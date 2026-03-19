use eframe::egui;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ── Chromium fling curve constants (fling_curve.cc) ───────────────────────────
const ALPHA: f64 = -5_707.62;
const BETA: f64 = 172.0;
const GAMMA: f64 = 3.7;

// ── Chromium FlingBooster constants (fling_booster.cc) ────────────────────────
const MIN_BOOST_FLING_SPEED_SQ: f64 = 350.0 * 350.0;
const MIN_BOOST_TOUCH_SCROLL_SPEED_SQ: f64 = 150.0 * 150.0;
const FLING_BOOST_TIMEOUT: Duration = Duration::from_millis(50);

// ── Chromium velocity tracker constants (velocity_tracker.cc) ─────────────────
const HISTORY_SIZE: usize = 20;
const HORIZON_MS: f64 = 100.0;

const MAGIC_SCROLL_FACTOR: f64 = 2.5;

// ─────────────────────────────────────────────────────────────────────────────
// Chromium curve math (fling_curve.cc)
// position(t) = α·e^(−γt) − β·t − α
// velocity(t) = −α·γ·e^(−γt) − β
// time_at_velocity(v) = −ln((v+β)/(−α·γ)) / γ
// ─────────────────────────────────────────────────────────────────────────────
fn position_at_time(t: f64) -> f64 {
    ALPHA * (-GAMMA * t).exp() - BETA * t - ALPHA
}
fn velocity_at_time(t: f64) -> f64 {
    -ALPHA * GAMMA * (-GAMMA * t).exp() - BETA
}
fn time_at_velocity(v: f64) -> f64 {
    -((v + BETA) / (-ALPHA * GAMMA)).ln() / GAMMA
}

// ─────────────────────────────────────────────────────────────────────────────
// LSQ2 velocity tracker — direct port of LeastSquaresVelocityTrackerStrategy
// (velocity_tracker.cc, degree=2, WEIGHTING_NONE, kHorizonMS=100)
//
// Chromium fits a degree-2 polynomial to the last 100ms of scroll positions
// via QR decomposition (Gram-Schmidt). Velocity = linear coefficient (b[1]).
// ─────────────────────────────────────────────────────────────────────────────

/// QR decomposition via Gram-Schmidt, solves R·b = Qᵀ·w·y.
/// Returns coefficients [b0, b1, b2, ...] or None if degenerate.
fn solve_least_squares(
    x: &[f64], // time values (negative age in seconds)
    y: &[f64], // position values
    w: &[f64], // weights (all 1.0 for WEIGHTING_NONE)
    m: usize,  // number of samples
    n: usize,  // degree + 1
) -> Option<Vec<f64>> {
    // Build A column-major, pre-multiplied by weights: a[col][row] = w[row] * x[row]^col
    let mut a = vec![vec![0.0f64; m]; n];
    for h in 0..m {
        a[0][h] = w[h];
        for i in 1..n {
            a[i][h] = a[i - 1][h] * x[h];
        }
    }

    // Gram-Schmidt orthonormalization → QR
    let mut q = vec![vec![0.0f64; m]; n];
    let mut r = vec![vec![0.0f64; n]; n];

    for j in 0..n {
        q[j] = a[j].clone();
        for i in 0..j {
            let dot: f64 = (0..m).map(|h| q[j][h] * q[i][h]).sum();
            for h in 0..m {
                q[j][h] -= dot * q[i][h];
            }
        }
        let norm: f64 = (0..m).map(|h| q[j][h] * q[j][h]).sum::<f64>().sqrt();
        if norm < 1e-6 {
            return None; // linearly dependent
        }
        let inv = 1.0 / norm;
        for h in 0..m {
            q[j][h] *= inv;
        }
        for i in 0..n {
            r[j][i] = if i < j {
                0.0
            } else {
                (0..m).map(|h| q[j][h] * a[i][h]).sum()
            };
        }
    }

    // Solve R·b = Qᵀ·(w·y)  (back-substitution)
    let wy: Vec<f64> = (0..m).map(|h| y[h] * w[h]).collect();
    let mut b = vec![0.0f64; n];
    for i in (0..n).rev() {
        b[i] = (0..m).map(|h| q[i][h] * wy[h]).sum();
        for j in (i + 1)..n {
            b[i] -= r[i][j] * b[j];
        }
        b[i] /= r[i][i];
    }
    Some(b)
}

struct LsqVelocityTracker {
    /// (timestamp, cumulative_scroll_position)
    samples: VecDeque<(Instant, f64)>,
    cumulative: f64,
}

impl LsqVelocityTracker {
    fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            cumulative: 0.0,
        }
    }

    fn push(&mut self, delta: f64) {
        self.cumulative += delta;
        let now = Instant::now();
        // Drop samples older than kHorizonMS
        self.samples
            .retain(|(t, _)| now.duration_since(*t).as_secs_f64() * 1000.0 < HORIZON_MS);
        self.samples.push_back((now, self.cumulative));
        while self.samples.len() > HISTORY_SIZE {
            self.samples.pop_front();
        }
    }

    /// Returns velocity in px/s using LSQ2 regression (same as Chromium default).
    fn velocity(&self) -> f64 {
        let m = self.samples.len();
        if m < 2 {
            return 0.0;
        }

        let newest_t = self.samples.back().unwrap().0;

        // time[] = negative age in seconds (newest = 0), y[] = cumulative position
        let times: Vec<f64> = self
            .samples
            .iter()
            .map(|(t, _)| -(newest_t.duration_since(*t).as_secs_f64()))
            .collect();
        let positions: Vec<f64> = self.samples.iter().map(|(_, p)| *p).collect();
        let weights = vec![1.0f64; m]; // WEIGHTING_NONE

        let degree = 2.min(m - 1);
        let n = degree + 1;

        match solve_least_squares(&times, &positions, &weights, m, n) {
            Some(b) if b.len() >= 2 => b[1], // b[1] = linear term = velocity
            _ => 0.0,
        }
    }

    fn clear(&mut self) {
        self.samples.clear();
        self.cumulative = 0.0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlingCurve (fling_curve.cc)
// ─────────────────────────────────────────────────────────────────────────────
struct FlingCurve {
    start: Instant,
    displacement_ratio: f64,
    time_offset: f64,
    position_offset: f64,
    curve_duration: f64,
    cumulative_scroll: f64,
    pub velocity: f64,
}

impl FlingCurve {
    fn new(velocity: f64) -> Self {
        let curve_duration = time_at_velocity(0.0);
        let max_start_speed = velocity.abs().min(velocity_at_time(0.0));
        let time_offset = time_at_velocity(max_start_speed);
        let position_offset = position_at_time(time_offset);
        Self {
            start: Instant::now(),
            displacement_ratio: velocity.signum(),
            time_offset,
            position_offset,
            curve_duration,
            cumulative_scroll: 0.0,
            velocity,
        }
    }

    /// Returns (delta_pixels, still_active) — mirrors ComputeScrollDeltaAtTime
    fn tick(&mut self) -> (f64, bool) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let offset_time = elapsed + self.time_offset;

        let (scalar_offset, still_active) = if offset_time < self.curve_duration {
            self.velocity = velocity_at_time(offset_time) * self.displacement_ratio;
            (position_at_time(offset_time) - self.position_offset, true)
        } else {
            self.velocity = 0.0;
            (
                position_at_time(self.curve_duration) - self.position_offset,
                false,
            )
        };

        let total = scalar_offset * self.displacement_ratio;
        let delta = total - self.cumulative_scroll;
        self.cumulative_scroll = total;
        (delta, still_active)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlingBooster (fling_booster.cc)
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Default)]
struct FlingBooster {
    pub previous_fling_starting_velocity: f64,
    current_fling_velocity: f64,
    cutoff_time_for_boost: Option<Instant>,
    previous_boosting_scroll_timestamp: Option<Instant>,
}

impl FlingBooster {
    fn get_velocity_for_fling_start(&mut self, new_velocity: f64) -> f64 {
        let velocity = if self.should_boost_fling(new_velocity) {
            new_velocity + self.previous_fling_starting_velocity
        } else {
            new_velocity
        };
        self.reset();
        self.previous_fling_starting_velocity = velocity;
        self.current_fling_velocity = velocity;
        velocity
    }

    fn observe_scroll_begin(&mut self) {
        if self.previous_fling_starting_velocity == 0.0 {
            return;
        }
        self.cutoff_time_for_boost = Some(Instant::now() + FLING_BOOST_TIMEOUT);
    }

    fn observe_scroll_update(&mut self, delta: f64) {
        if self.previous_fling_starting_velocity == 0.0 {
            return;
        }
        if self.cutoff_time_for_boost.is_some_and(|t| Instant::now() > t) {
            self.reset();
            return;
        }
        if self.cutoff_time_for_boost.is_none() {
            return;
        }
        // Counter-direction kills boost
        if (delta >= 0.0) != (self.previous_fling_starting_velocity >= 0.0) {
            self.reset();
            return;
        }
        // Scroll must be fast enough to maintain the fling
        if let Some(prev_t) = self.previous_boosting_scroll_timestamp {
            let dt = prev_t.elapsed().as_secs_f64();
            if dt >= 0.001 {
                let sv = delta / dt;
                if sv * sv < MIN_BOOST_TOUCH_SCROLL_SPEED_SQ {
                    self.reset();
                    return;
                }
            }
        }
        self.previous_boosting_scroll_timestamp = Some(Instant::now());
        self.cutoff_time_for_boost = Some(Instant::now() + FLING_BOOST_TIMEOUT);
    }

    fn observe_scroll_end(&mut self) {
        self.previous_boosting_scroll_timestamp = None;
    }

    fn observe_fling_cancel(&mut self, prevent_boosting: bool) {
        if prevent_boosting {
            self.reset();
            return;
        }
        self.previous_boosting_scroll_timestamp = None;
        self.cutoff_time_for_boost = Some(Instant::now() + FLING_BOOST_TIMEOUT);
    }

    fn observe_progress_fling(&mut self, current_velocity: f64) {
        if self.previous_fling_starting_velocity == 0.0 {
            return;
        }
        self.current_fling_velocity = current_velocity;
    }

    fn should_boost_fling(&self, new_velocity: f64) -> bool {
        if self.previous_fling_starting_velocity == 0.0 {
            return false;
        }
        if self.cutoff_time_for_boost.is_none() {
            return false;
        }
        if self.cutoff_time_for_boost.is_some_and(|t| Instant::now() > t) {
            return false;
        }
        if (new_velocity >= 0.0) != (self.previous_fling_starting_velocity >= 0.0) {
            return false;
        }
        if self.current_fling_velocity * self.current_fling_velocity < MIN_BOOST_FLING_SPEED_SQ {
            return false;
        }
        if new_velocity * new_velocity < MIN_BOOST_FLING_SPEED_SQ {
            return false;
        }
        true
    }

    pub fn reset(&mut self) {
        self.cutoff_time_for_boost = None;
        self.previous_fling_starting_velocity = 0.0;
        self.current_fling_velocity = 0.0;
        self.previous_boosting_scroll_timestamp = None;
    }
}

fn wheel_detent_step(page_size: f64) -> f64 {
    page_size.powf(2.0 / 3.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// App
// ─────────────────────────────────────────────────────────────────────────────
const ITEM_H: f32 = 60.0;
const ITEM_N: usize = 500;
const CONTENT_H: f64 = ITEM_H as f64 * ITEM_N as f64;

struct App {
    offset: f64,
    fling: Option<FlingCurve>,
    booster: FlingBooster,
    tracker: LsqVelocityTracker, // ← LSQ2, same as Chromium
    last_scroll: Option<Instant>,
    scroll_began: bool, // tracks GestureScrollBegin equivalent
    is_trackpad: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            offset: 0.0,
            fling: None,
            booster: FlingBooster::default(),
            tracker: LsqVelocityTracker::new(),
            last_scroll: None,
            scroll_began: false,
            is_trackpad: false,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            let view_h = rect.height() as f64;
            let max_offset = (CONTENT_H - view_h).max(0.0);
            let step = wheel_detent_step(view_h);

            let (raw_delta, is_trackpad) = ctx.input(|i| {
                let trackpad = i.smooth_scroll_delta != i.raw_scroll_delta;
                (i.raw_scroll_delta.y, trackpad)
            });
            self.is_trackpad = is_trackpad;

            if raw_delta != 0.0 {
                // GestureFlingCancel (finger down on active fling) → open boost window
                if self.fling.is_some() {
                    self.booster.observe_fling_cancel(false);
                    self.fling = None;
                }

                // GestureScrollBegin on first event of a new gesture
                if !self.scroll_began {
                    self.booster.observe_scroll_begin();
                    self.scroll_began = true;
                }

                let scaled = if is_trackpad {
                    -raw_delta as f64 * MAGIC_SCROLL_FACTOR
                } else {
                    -raw_delta as f64 / 50.0 * step
                };

                // GestureScrollUpdate
                self.booster.observe_scroll_update(scaled);
                self.tracker.push(scaled);
                self.last_scroll = Some(Instant::now());
                self.offset = (self.offset + scaled).clamp(0.0, max_offset);
                ctx.request_repaint();
            } else if self.last_scroll.is_some_and(|t| t.elapsed().as_millis() > 30) {
                // GestureScrollEnd → launch fling
                self.scroll_began = false;
                let raw_vel = self.tracker.velocity();

                // kMinInertialScrollDelta equivalent (> 0.1 handled in fling tick)
                if raw_vel.abs() > 20.0 {
                    self.booster.observe_scroll_end();
                    let vel = self.booster.get_velocity_for_fling_start(raw_vel);
                    self.fling = Some(FlingCurve::new(vel));
                } else {
                    self.booster.reset();
                }

                self.last_scroll = None;
                self.tracker.clear();
            }

            // ProgressFling — called every BeginFrame (vsync), we use request_repaint
            if let Some(f) = &mut self.fling {
                let (delta, running) = f.tick();
                self.booster.observe_progress_fling(f.velocity);

                // kMinInertialScrollDelta = 0.1 (fling_controller.cc)
                if delta.abs() > 0.1 {
                    self.offset = (self.offset + delta).clamp(0.0, max_offset);
                }

                if !running {
                    self.booster.reset();
                    self.fling = None;
                }
                ctx.request_repaint();
            }

            // ── draw ──────────────────────────────────────────────────────────
            let painter = ui.painter_at(rect);
            ui.set_clip_rect(rect);

            for i in 0..ITEM_N {
                let y = i as f32 * ITEM_H - self.offset as f32;
                if y + ITEM_H < 0.0 || y > rect.height() {
                    continue;
                }
                let item_rect = egui::Rect::from_min_size(
                    egui::pos2(rect.left() + 8.0, rect.top() + y + 4.0),
                    egui::vec2(rect.width() - 16.0, ITEM_H - 8.0),
                );
                let t = i as f32 / ITEM_N as f32;
                let color = egui::Color32::from_rgb(
                    (60.0 + t * 80.0) as u8,
                    (100.0 + t * 40.0) as u8,
                    (180.0 - t * 60.0) as u8,
                );
                painter.rect_filled(item_rect, 6.0, color);
                painter.text(
                    item_rect.left_center() + egui::vec2(14.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    format!("Item {:03}", i + 1),
                    egui::FontId::proportional(16.0),
                    egui::Color32::WHITE,
                );
            }

            // scrollbar
            if max_offset > 0.0 {
                let thumb_h = (view_h / CONTENT_H * rect.height() as f64) as f32;
                let thumb_y =
                    (self.offset / max_offset) as f32 * (rect.height() - thumb_h);
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(rect.right() - 6.0, rect.top() + thumb_y),
                        egui::vec2(4.0, thumb_h),
                    ),
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(160, 160, 160, 200),
                );
            }

            // debug
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(format!(
                    "offset: {:.0}  fling_vel: {:.0}  boost_prev: {:.0}  input: {}",
                    self.offset,
                    self.fling.as_ref().map_or(0.0, |f| f.velocity),
                    self.booster.previous_fling_starting_velocity,
                    if self.is_trackpad { "trackpad" } else { "wheel" },
                ));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Kinetic Scroll — Chromium 1:1",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}