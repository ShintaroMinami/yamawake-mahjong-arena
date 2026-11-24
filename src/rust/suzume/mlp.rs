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
//! The rows of one decision go through together (`pre_rows`, `activate_rows`):
//! each weight column is read once for all of them, and every row's values
//! are those of the row alone.
//!
//! The same layers in f32 (`pre_rows_f32`, `activate_rows_f32`; the A1
//! option `mlp=f32`, 2026-11-24) run the same loop on f32 copies of the
//! weights: faster, the values within f32 rounding of the f64 ones.

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
    /// `w1t` in f32 (`pre_rows_f32`).
    w1t32: Vec<f32>,
    /// `deep_t` and the further layers' biases in f32 (`activate_rows_f32`).
    deep32: Vec<(Vec<f32>, Vec<f32>)>,
}

/// The start value of `Iterator::sum` over `f64` (kept so that the column
/// sums match a per-unit `sum` bit for bit, signed zeros included).
fn sum_start() -> f64 {
    std::iter::empty::<f64>().sum()
}

/// For every row `r`: `acc[r][k] += sum_j wt[j][k] x[r][j]` in `j` order;
/// `acc` holds rows of `h` and `x` the same number of rows of `n_in`, `wt`
/// `n_in` columns of `h`. A zero input's column is skipped (after a ReLU
/// about half of them): its terms are signed zeros, which leave every
/// nonzero sum bit for bit as it is and can only flip the sign of a sum that
/// is exactly zero, a difference nothing downstream reads (comparisons,
/// `exp`, ReLU and further sums treat both zeros alike). Column `j` is
/// applied to every row before column `j + 1`, so it is read once for all
/// rows; each row's sums are those of the row alone. The rows with a nonzero
/// input in a column are found from a bit mask per column (64 rows at a
/// time), so the skips cost no unpredictable branch per input. `T` is f64,
/// or f32 for `Precision::F32` (`pre_rows_f32`; `T::default()` is the zero).
fn add_columns<T>(acc: &mut [T], h: usize, wt: &[T], x: &[T], n_in: usize)
where
    T: Copy + Default + PartialEq + std::ops::Mul<Output = T> + std::ops::AddAssign,
{
    if n_in == 0 {
        return;
    }
    for (acc, x) in acc.chunks_mut(64 * h).zip(x.chunks(64 * n_in)) {
        let mut mask = vec![0u64; n_in];
        for (r, xr) in x.chunks_exact(n_in).enumerate() {
            for (m, &v) in mask.iter_mut().zip(xr) {
                *m |= u64::from(v != T::default()) << r;
            }
        }
        for (j, (col, &m)) in wt.chunks_exact(h).zip(&mask).enumerate() {
            let mut m = m;
            while m != 0 {
                let r = m.trailing_zeros() as usize;
                m &= m - 1;
                let xj = x[r * n_in + j];
                for (a, &w) in acc[r * h..(r + 1) * h].iter_mut().zip(col) {
                    *a += w * xj;
                }
            }
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
        self.pre_rows(base, x, 1, from)
    }

    /// `pre_from` of each of the `n_rows` rows of `xs` (row-major), the rows'
    /// pre-activations one after the other.
    pub fn pre_rows(&self, base: &[f64], xs: &[f64], n_rows: usize, from: usize) -> Vec<f64> {
        let (h, width) = (self.widths[0], xs.len() / n_rows.max(1));
        let mut acc = vec![sum_start(); n_rows * h];
        add_columns(&mut acc, h, &self.w1t[from * h..(from + width) * h], xs, width);
        for row in acc.chunks_exact_mut(h) {
            for (s, b) in row.iter_mut().zip(base) {
                *s = b + *s;
            }
        }
        acc
    }

    /// The last layer's activations from the first layer's pre-activations.
    pub fn activate(&self, z: Vec<f64>) -> Vec<f64> {
        self.activate_rows(z)
    }

    /// `activate` of each row of `z` (rows of the first layer's width,
    /// row-major), the rows' activations one after the other.
    pub fn activate_rows(&self, mut z: Vec<f64>) -> Vec<f64> {
        for v in &mut z {
            *v = v.max(0.0);
        }
        let mut width = self.widths[0];
        for ((_, b), wt) in self.deep.iter().zip(&self.deep_t) {
            let mut acc = vec![sum_start(); z.len() / width * b.len()];
            add_columns(&mut acc, b.len(), wt, &z, width);
            for row in acc.chunks_exact_mut(b.len()) {
                for (s, bk) in row.iter_mut().zip(b) {
                    *s = (bk + *s).max(0.0);
                }
            }
            (z, width) = (acc, b.len());
        }
        z
    }

    /// `pre_rows` in f32 from the first input column (`from` = 0): every row
    /// starts from `base` (the decision's own first-layer terms, `b1`
    /// included).
    pub fn pre_rows_f32(&self, base: &[f32], xs: &[f32], n_rows: usize) -> Vec<f32> {
        let (h, width) = (self.widths[0], xs.len() / n_rows.max(1));
        let mut acc: Vec<f32> = (0..n_rows).flat_map(|_| base.iter().copied()).collect();
        add_columns(&mut acc, h, &self.w1t32[..width * h], xs, width);
        acc
    }

    /// `activate_rows` in f32.
    pub fn activate_rows_f32(&self, mut z: Vec<f32>) -> Vec<f32> {
        for v in &mut z {
            *v = v.max(0.0);
        }
        let mut width = self.widths[0];
        for (wt, b) in &self.deep32 {
            let n_rows = z.len() / width;
            let mut acc: Vec<f32> = (0..n_rows).flat_map(|_| b.iter().copied()).collect();
            add_columns(&mut acc, b.len(), wt, &z, width);
            for v in &mut acc {
                *v = v.max(0.0);
            }
            (z, width) = (acc, b.len());
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
    let deep_t: Vec<Vec<f64>> = deep.iter().zip(&widths).map(|((w, b), &c)| transpose(w, b.len(), c)).collect();
    let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let w1t32 = f32s(&w1t);
    let deep32 = deep_t.iter().zip(&deep).map(|(wt, (_, b))| (f32s(wt), f32s(b))).collect();
    Ok(Hidden { n_cols, w1, b1, deep, widths, w1t, deep_t, w1t32, deep32 })
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

    #[test]
    fn rows_together_equal_each_row_alone() {
        // Pseudo-random weights and rows (some inputs exactly zero), against the plain per-unit dot products.
        let mut seed = 7u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let (n_in, widths) = (9, [7, 5, 6]);
        let mut mat = |r: usize, c: usize| -> Vec<Vec<f64>> { (0..r).map(|_| (0..c).map(|_| rnd()).collect()).collect() };
        let (w1, b1) = (mat(widths[0], n_in), mat(1, widths[0]).remove(0));
        let deep: Vec<(Vec<Vec<f64>>, Vec<f64>)> = (1..widths.len()).map(|i| (mat(widths[i], widths[i - 1]), mat(1, widths[i]).remove(0))).collect();
        // More than 64 rows: the rows go 64 at a time (`add_columns`).
        let mut xs = mat(70, n_in);
        xs[1][3] = 0.0;
        xs[4] = vec![0.0; n_in];
        xs[66][0] = 0.0;
        let v = serde_json::json!({"hidden": widths[0], "W1": w1, "b1": b1,
                                   "Wh": deep.iter().map(|d| &d.0).collect::<Vec<_>>(), "bh": deep.iter().map(|d| &d.1).collect::<Vec<_>>()});
        let net = parse_hidden(&v, n_in).unwrap();
        let dense = |w: &[Vec<f64>], b: &[f64], x: &[f64]| -> Vec<f64> {
            w.iter().zip(b).map(|(r, bk)| bk + r.iter().zip(x).map(|(a, c)| a * c).sum::<f64>()).collect()
        };
        let want: Vec<f64> = xs
            .iter()
            .flat_map(|x| {
                let mut h: Vec<f64> = dense(&w1, &b1, x).iter().map(|z| z.max(0.0)).collect();
                for (w, b) in &deep {
                    h = dense(w, b, &h).iter().map(|z| z.max(0.0)).collect();
                }
                h
            })
            .collect();
        let flat: Vec<f64> = xs.concat();
        let got = net.activate_rows(net.pre_rows(&net.b1, &flat, xs.len(), 0));
        assert_eq!(got, want);
        assert!(got.iter().any(|&g| g > 0.0));
        let one: Vec<f64> = xs.iter().flat_map(|x| net.activate(net.pre_dense(x))).collect();
        assert_eq!(one, want);
        // The f32 layers (`Precision::F32`) on the same rows: within f32 rounding of the f64 values.
        let flat32: Vec<f32> = flat.iter().map(|&x| x as f32).collect();
        let b32: Vec<f32> = net.b1.iter().map(|&b| b as f32).collect();
        let got32 = net.activate_rows_f32(net.pre_rows_f32(&b32, &flat32, xs.len()));
        assert_eq!(got32.len(), want.len());
        for (g, w) in got32.iter().zip(&want) {
            assert!((f64::from(*g) - w).abs() <= 1e-5 * (1.0 + w.abs()), "{g} vs {w}");
        }
    }
}
