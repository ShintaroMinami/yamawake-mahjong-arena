//! One decision's context: the view, the config, and the memo tables the
//! rules share (shanten/ukeire, the legal discard candidates, DP tables).
//!
//! Everything is computed on first use and kept for the rest of the
//! decision, so a rule pays only for what it reads.

use std::cell::{OnceCell, RefCell};
use std::cmp::Ordering;
use std::rc::Rc;

use riichienv_core::action::{Action, ActionType};

use super::config::SuzumeConfig;
use super::memo::SeatMemo;
use super::params::{Models, set_models};
use super::sp::{DpRow, dp_input, dp_rows_of};
use super::universal::{Universal, universal};
use super::ukeire::UkeireTables;
use super::view::MjView;
use crate::eval::feat::tiles::is_aka_tid;

/// One distinct legal discard: the tile and the shanten of what is left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiscardCand {
    pub tile_type: u8,
    pub is_aka: bool,
    pub shanten: i8,
}

pub struct Ctx<'a> {
    pub view: &'a MjView,
    pub cfg: &'a SuzumeConfig,
    /// The seat's model set (`cfg.models`).
    pub models: &'static Models,
    tables: RefCell<UkeireTables>,
    cands: OnceCell<Vec<DiscardCand>>,
    /// DP tables with riichi assumed (`[1]`) or not (`[0]`).
    dp: [OnceCell<Option<Vec<DpRow>>>; 2],
    /// The decision's universal rows, keyed by `change`.
    universal: RefCell<Vec<(bool, Rc<Universal>)>>,
    /// The universal MLPs' option scores, keyed by the model (its address)
    /// and the universal rows they scored (`universal_mlp::mlp_choice_in`).
    mlp_scores: RefCell<Vec<(usize, Rc<Universal>, Vec<f64>)>>,
    /// The seat's per-kyoku memo, lent to this decision.
    memo: RefCell<SeatMemo>,
}

impl<'a> Ctx<'a> {
    /// A context with an empty memo.
    #[cfg(test)]
    pub fn new(view: &'a MjView, cfg: &'a SuzumeConfig) -> Self {
        Self::with_memo(view, cfg, SeatMemo::default())
    }

    /// A context using (and filling) the seat's memo; `into_memo` returns it.
    pub fn with_memo(view: &'a MjView, cfg: &'a SuzumeConfig, memo: SeatMemo) -> Self {
        Self {
            view,
            cfg,
            models: set_models(cfg.models),
            tables: RefCell::new(UkeireTables::new(view.unseen())),
            cands: OnceCell::new(),
            dp: [OnceCell::new(), OnceCell::new()],
            universal: RefCell::new(Vec::new()),
            mlp_scores: RefCell::new(Vec::new()),
            memo: RefCell::new(memo),
        }
    }

    /// The memo, back from this decision.
    pub fn into_memo(self) -> SeatMemo {
        self.memo.into_inner()
    }

    /// Lend the memo out (to a nested context); `put_memo` returns it.
    pub fn take_memo(&self) -> SeatMemo {
        std::mem::take(&mut *self.memo.borrow_mut())
    }

    pub fn put_memo(&self, memo: SeatMemo) {
        *self.memo.borrow_mut() = memo;
    }

    /// The memo, borrowed.
    pub fn memo(&self) -> std::cell::RefMut<'_, SeatMemo> {
        self.memo.borrow_mut()
    }

    /// The shared shanten/ukeire tables.
    pub fn tables(&self) -> std::cell::RefMut<'_, UkeireTables> {
        self.tables.borrow_mut()
    }

    /// Every distinct legal discard (tile type, aka), in tile order.
    pub fn discard_cands(&self) -> &[DiscardCand] {
        self.cands.get_or_init(|| {
            let mut out: Vec<DiscardCand> = Vec::with_capacity(14);
            for a in &self.view.legal {
                let (ActionType::Discard, Some(tid)) = (a.action_type, a.tile) else { continue };
                let (tile_type, is_aka) = (tid / 4, is_aka_tid(tid));
                if out.iter().any(|c| c.tile_type == tile_type && c.is_aka == is_aka) {
                    continue;
                }
                let mut h = self.view.hand;
                h[tile_type as usize] -= 1;
                let shanten = self.tables().shanten(&h);
                out.push(DiscardCand { tile_type, is_aka, shanten });
            }
            out.sort_by_key(|c| (c.tile_type, c.is_aka));
            out
        })
    }

    /// The lowest shanten a legal discard leaves, if any discard is legal.
    pub fn min_shanten(&self) -> Option<i8> {
        self.discard_cands().iter().map(|c| c.shanten).min()
    }

    /// The DP table of this turn's discards (`None` when no draw is left or
    /// the hand holds a meld), solved once per assumption, ron counted at
    /// `params().ron_rate` (rows carry `ron_win`, `dama_ron_possible`,
    /// `furiten`; through the seat's solve cache, `sp::SolveCache`).
    pub fn dp(&self, assume_riichi: bool) -> Option<&[DpRow]> {
        self.dp[usize::from(assume_riichi)]
            .get_or_init(|| Some(dp_rows_of(&dp_input(self.view, assume_riichi)?, &mut self.memo())))
            .as_deref()
    }

    /// This decision's options and U rows (`universal::universal`), computed
    /// once for the rules that read them (the push / fold features' hand MLP,
    /// the kan MLP's keep, the universal MLP).
    pub fn universal(&self, change: bool) -> Rc<Universal> {
        if let Some((_, u)) = self.universal.borrow().iter().find(|(k, _)| *k == change) {
            return Rc::clone(u);
        }
        let u = Rc::new(universal(self, change));
        self.universal.borrow_mut().push((change, Rc::clone(&u)));
        u
    }

    /// Model `model`'s scores of the rows `u` (`score` on the first call,
    /// then kept for the rest of the decision).
    pub fn mlp_scores(&self, model: usize, u: &Rc<Universal>, score: impl FnOnce() -> Vec<f64>) -> Vec<f64> {
        let hit = |(m, v, _): &&(usize, Rc<Universal>, Vec<f64>)| *m == model && Rc::ptr_eq(v, u);
        if let Some((_, _, s)) = self.mlp_scores.borrow().iter().find(hit) {
            return s.clone();
        }
        let s = score();
        self.mlp_scores.borrow_mut().push((model, Rc::clone(u), s.clone()));
        s
    }

    /// The legal discard of (tile type, aka); the drawn tile itself when it
    /// is that tile (tsumogiri), else a copy from the hand.
    pub fn discard_action(&self, tile_type: u8, is_aka: bool) -> Option<Action> {
        let drawn = self.view.drawn_tile;
        let mut pick: Option<&Action> = None;
        for a in &self.view.legal {
            let (ActionType::Discard, Some(tid)) = (a.action_type, a.tile) else { continue };
            if tid / 4 != tile_type || is_aka_tid(tid) != is_aka {
                continue;
            }
            if Some(tid) == drawn {
                return Some(a.clone());
            }
            pick.get_or_insert(a);
        }
        pick.cloned()
    }

    /// Order two discards the deciding rule left tied: the lower tile type,
    /// then the plain copy before the red one.
    pub fn tie_break(&self, a: &DiscardCand, b: &DiscardCand) -> Ordering {
        (a.tile_type, a.is_aka).cmp(&(b.tile_type, b.is_aka))
    }

    /// Riichi is assumed once tenpai: already declared, or affordable and
    /// riichi is on.
    pub fn assume_riichi(&self) -> bool {
        self.view.in_riichi || (self.cfg.riichi && self.view.score >= 1000)
    }
}
