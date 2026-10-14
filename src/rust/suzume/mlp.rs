//! The hidden layers every suzume MLP shares (`Umlp`, `FoldMlp`,
//! `StateMlp`): a first ReLU layer `W1` / `b1` over the model's inputs, then
//! any number of further ReLU layers (`"Wh"` / `"bh"` in the file: one matrix
//! and one bias per layer; none in a one-layer model). The models differ only
//! in the first layer's inputs (each builds its pre-activations, e.g. adding a
//! state one-hot column) and in the head after the last layer.
//!
//! ```text
//! z1 = W1 x + b1                     (model-specific: `pre_dense`, `w1_row`)
//!      [+ W1[:, sit] s]               (a file with `sit_inputs`: the decision's
//!                                     situation, `pf::situation`, standardized
//!                                     by `sit_mean` / `sit_std`; `Sit::bias`)
//! h1 = relu(z1)
//! h_{i+1} = relu(Wh[i] h_i + bh[i])  (`activate`)
//! ```
//!
//! The products are summed column by column over every unit at once (the
//! weights are also kept transposed, so a column is contiguous), which
//! vectorizes; each unit still adds its terms in input order from the same
//! start as `Iterator::sum`, so the values are bit-identical to the plain
//! dot product per unit (up to the sign of an exact zero, `add_columns`).

use serde_json::Value;

use super::view::MjView;

/// The situation inputs of a model file (`"sit_inputs"`, the names of
/// `pf::SIT_NAMES` in that order; the last columns of the first layer).
#[derive(Debug)]
pub struct Sit {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    /// The first of the situation columns of `W1`.
    pub col: usize,
}

impl Sit {
    /// Each first-layer unit's share from the situation of view `v` (the
    /// same for every row of a decision).
    pub fn bias(&self, net: &Hidden, v: &MjView) -> Vec<f64> {
        let s: Vec<f64> = crate::suzume::pf::situation(v).iter().enumerate().map(|(i, x)| (x - self.mean[i]) / self.std[i]).collect();
        net.pre_from(&vec![0.0; net.widths[0]], &s, self.col)
    }
}

/// The situation inputs of a model file, when it has `sit_inputs`; they
/// follow the model's own `n_own` first-layer columns.
pub fn parse_sit(v: &Value, n_own: usize) -> Result<Option<Sit>, String> {
    let Some(names) = v.get("sit_inputs") else { return Ok(None) };
    let names: Vec<&str> = names.as_array().ok_or("sit_inputs is not a list")?.iter().map(|x| x.as_str().unwrap_or("")).collect();
    if names != crate::suzume::pf::SIT_NAMES {
        return Err(format!("sit_inputs must be {:?}", crate::suzume::pf::SIT_NAMES));
    }
    let stat = |k: &str| -> Result<Vec<f64>, String> { vector(v.get(k).ok_or_else(|| format!("missing {k}"))?, k) };
    let (mean, std) = (stat("sit_mean")?, stat("sit_std")?);
    if mean.len() != names.len() || std.len() != names.len() || std.iter().any(|&x| x <= 0.0) {
        return Err("sit_mean / sit_std do not match sit_inputs".into());
    }
    Ok(Some(Sit { mean, std, col: n_own }))
}

/// Adds a decision's situation share (`Sit::bias`; empty: none) to the first
/// layer's pre-activations.
pub fn add_bias(z: &mut [f64], sit: &[f64]) {
    for (a, b) in z.iter_mut().zip(sit) {
        *a += b;
    }
}

/// The hidden layers of one model.
#[derive(Debug)]
pub struct Hidden {
    /// The first layer's input width (all its columns).
    pub n_cols: usize,
    /// The first layer: `widths[0]` rows of `n_cols`, row-major.
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    /// The further layers: layer `i` has `widths[i + 1]` rows of `widths[i]`
    /// (row-major) and its bias.
    pub deep: Vec<(Vec<f64>, Vec<f64>)>,
    /// The output width of every layer, the first first.
    pub widths: Vec<usize>,
    /// `w1` transposed: `n_cols` columns of `widths[0]`.
    w1t: Vec<f64>,
    /// Each further layer's matrix transposed (its input columns of its
    /// output width).
    deep_t: Vec<Vec<f64>>,
}

/// The start value of `Iterator::sum` over `f64` (kept so that the column
/// sums match a per-unit `sum` bit for bit, signed zeros included).
fn sum_start() -> f64 {
    std::iter::empty::<f64>().sum()
}

/// `acc[k] += sum_j wt[j][k] x[j]` in `j` order; `wt` holds `x.len()`
/// columns of `acc.len()`. A zero input's column is skipped (after a ReLU
/// about half of them): its terms are signed zeros, which leave every
/// nonzero sum bit for bit as it is and can only flip the sign of a sum that
/// is exactly zero, a difference nothing downstream reads (comparisons,
/// `exp`, ReLU and further sums treat both zeros alike).
fn add_columns(acc: &mut [f64], wt: &[f64], x: &[f64]) {
    let n = acc.len();
    for (col, &xj) in wt.chunks_exact(n).zip(x) {
        if xj == 0.0 {
            continue;
        }
        for (a, &w) in acc.iter_mut().zip(col) {
            *a += w * xj;
        }
    }
}

/// A row-major `rows x cols` matrix transposed.
fn transpose(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    (0..cols * rows).map(|i| m[(i % rows) * cols + i / rows]).collect()
}

impl Hidden {
    /// The width of the last layer (the head's input).
    pub fn out_width(&self) -> usize {
        *self.widths.last().expect("at least one layer")
    }

    /// The first layer's weights of unit `k` (all `n_cols` columns).
    pub fn w1_row(&self, k: usize) -> &[f64] {
        &self.w1[k * self.n_cols..(k + 1) * self.n_cols]
    }

    /// The first layer's pre-activations from the leading inputs `x` (the
    /// remaining columns contribute nothing): `b1 + W1[:, ..x.len()] x`.
    pub fn pre_dense(&self, x: &[f64]) -> Vec<f64> {
        self.pre_from(&self.b1, x, 0)
    }

    /// `base[k] + W1[k, from..from + x.len()] . x` for every first-layer
    /// unit `k` (`pre_dense` is `base` = `b1`, `from` = 0).
    pub fn pre_from(&self, base: &[f64], x: &[f64], from: usize) -> Vec<f64> {
        let h = self.widths[0];
        let mut acc = vec![sum_start(); h];
        add_columns(&mut acc, &self.w1t[from * h..(from + x.len()) * h], x);
        for (s, b) in acc.iter_mut().zip(base) {
            *s = b + *s;
        }
        acc
    }

    /// The last layer's activations from the first layer's pre-activations.
    pub fn activate(&self, mut z: Vec<f64>) -> Vec<f64> {
        for v in &mut z {
            *v = v.max(0.0);
        }
        for ((_, b), wt) in self.deep.iter().zip(&self.deep_t) {
            let mut acc = vec![sum_start(); b.len()];
            add_columns(&mut acc, wt, &z);
            for (s, bk) in acc.iter_mut().zip(b) {
                *s = (bk + *s).max(0.0);
            }
            z = acc;
        }
        z
    }
}

/// A list of equal-length number rows: (rows, columns, the numbers row-major).
pub fn matrix(v: &Value, what: &str) -> Result<(usize, usize, Vec<f64>), String> {
    let rows = v.as_array().ok_or_else(|| format!("{what} is not a list"))?;
    let mut out = Vec::new();
    let mut width = None;
    for row in rows {
        let row = row.as_array().ok_or_else(|| format!("{what} row is not a list"))?;
        if *width.get_or_insert(row.len()) != row.len() {
            return Err(format!("{what} rows differ in length"));
        }
        for x in row {
            out.push(x.as_f64().ok_or_else(|| format!("{what} holds a non-number"))?);
        }
    }
    Ok((rows.len(), width.unwrap_or(0), out))
}

/// A list of numbers.
pub fn vector(v: &Value, what: &str) -> Result<Vec<f64>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{what} holds a non-number")))
        .collect()
}

/// The hidden layers of a model file whose first layer reads `n_cols` inputs:
/// `W1` (`hidden` rows), `b1`, and the optional `Wh` / `bh`.
pub fn parse_hidden(v: &Value, n_cols: usize) -> Result<Hidden, String> {
    let field = |k: &str| v.get(k).ok_or_else(|| format!("missing {k}"));
    let (rows, cols, w1) = matrix(field("W1")?, "W1")?;
    let hidden = field("hidden")?.as_f64().ok_or("hidden is not a number")? as usize;
    let b1 = vector(field("b1")?, "b1")?;
    if rows != hidden || cols != n_cols || b1.len() != hidden {
        return Err(format!("W1 / b1 must be {hidden} x {n_cols} / {hidden}"));
    }
    let mut widths = vec![hidden];
    let mut deep = Vec::new();
    if let Some(wh) = v.get("Wh") {
        let wh = wh.as_array().ok_or("Wh is not a list")?;
        let bh = v.get("bh").and_then(Value::as_array).ok_or("Wh needs bh, a list")?;
        if wh.len() != bh.len() {
            return Err("Wh and bh differ in length".into());
        }
        for (i, (w, b)) in wh.iter().zip(bh).enumerate() {
            let (r, c, w) = matrix(w, &format!("Wh[{i}]"))?;
            let b = vector(b, &format!("bh[{i}]"))?;
            if c != *widths.last().expect("first layer") || b.len() != r || r == 0 {
                return Err(format!("Wh[{i}] / bh[{i}] do not follow the previous layer"));
            }
            widths.push(r);
            deep.push((w, b));
        }
    }
    let w1t = transpose(&w1, hidden, n_cols);
    let deep_t = deep.iter().zip(&widths).map(|((w, b), &c)| transpose(w, b.len(), c)).collect();
    Ok(Hidden { n_cols, w1, b1, deep, widths, w1t, deep_t })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_layers_follow_the_first() {
        let v: Value = serde_json::from_str(
            r#"{"hidden": 2, "W1": [[1, 0, 0], [0, -1, 0]], "b1": [0, 0],
                "Wh": [[[1, 1], [1, -1]]], "bh": [[0.5, 0]]}"#,
        )
        .unwrap();
        let h = parse_hidden(&v, 3).unwrap();
        assert_eq!(h.widths, [2, 2]);
        // x = (2, -3): z1 = (2, 3), h2 = relu((5.5, -1)).
        assert_eq!(h.activate(h.pre_dense(&[2.0, -3.0])), [5.5, 0.0]);
        let bad: Value = serde_json::from_str(r#"{"hidden": 2, "W1": [[1, 0], [0, 1]], "b1": [0, 0], "Wh": [[[1, 1, 1]]], "bh": [[0]]}"#).unwrap();
        assert!(parse_hidden(&bad, 2).is_err());
    }
}
