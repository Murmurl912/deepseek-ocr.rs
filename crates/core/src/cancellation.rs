use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// Cooperative cancellation flag shared across decoding layers.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a fresh token in the "not cancelled" state.
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation. Subsequent `is_cancelled` calls return true.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Returns true once `cancel` has been called.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}
