use eframe::egui;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

// Chromium fling curve constants (fling_curve.cc)
const ALPHA: f64 = -5_707.62;
const BETA: f64 = 172.0;
const GAMMA: f64 = 3.7;

//  PhysicsBasedFlingCurve constants (physics_based_fling_curve.cc)
const P1X: f64 = 0.2;
const P1Y: f64 = 1.0;
const P2X: f64 = 0.55;
const P2Y: f64 = 1.0;
const DEFAULT_PIXEL_DECELERATION: f64 = 2300.0;        // px/s  (used for duration)
const MAX_CURVE_DURATION: f64 = 2.0;                   // seconds
const PHYSICAL_DECELERATION: f64 = 2.7559e-5;          // inch/ms²
const DEFAULT_PIXELS_PER_INCH: f64 = 96.0;             // kDefaultPixelsPerInch
const DEFAULT_BOUNDS_MULTIPLIER: f64 = 3.0;            // from physics_based_fling_curve.h

// ── FlingBooster constants (fling_booster.cc) ──────────────────────────────
const MIN_BOOST_FLING_SPEED_SQ: f64 = 350.0 * 350.0;
const MIN_BOOST_TOUCH_SCROLL_SPEED_SQ: f64 = 150.0 * 150.0;
const FLING_BOOST_TIMEOUT: Duration = Duration::from_millis(50);

// ── VelocityTracker constants (velocity_tracker.cc) ────────────────────────
const HISTORY_SIZE: usize = 20;
const HORIZON_MS: f64 = 100.0;

const MAGIC_SCROLL_FACTOR: f64 = 2.5;

// ──────────────────────────────────────────────────────────────────────────
// Chromium FlingCurve math (fling_curve.cc)
// ──────────────────────────────────────────────────────────────────────────
fn fc_position(t: f64) -> f64 { ALPHA * (-GAMMA * t).exp() - BETA * t - ALPHA }
fn fc_velocity(t: f64) -> f64 { -ALPHA * GAMMA * (-GAMMA * t).exp() - BETA }
fn fc_time_at_vel(v: f64) -> f64 { -((v + BETA) / (-ALPHA * GAMMA)).ln() / GAMMA }

// ──────────────────────────────────────────────────────────────────────────
// CSS cubic-bezier solver — used by PhysicsBasedFlingCurve
// Same algorithm as gfx::CubicBezier in Chromium (Newton + bisection fallback)
// ──────────────────────────────────────────────────────────────────────────
struct CubicBezier { cx: f64, bx: f64, ax: f64, cy: f64, by: f64, ay: f64 }

impl CubicBezier {
    fn new(p1x: f64, p1y: f64, p2x: f64, p2y: f64) -> Self {
        let cx = 3.0 * p1x;
        let bx = 3.0 * (p2x - p1x) - cx;
        let ax = 1.0 - cx - bx;
        let cy = 3.0 * p1y;
        let by = 3.0 * (p2y - p1y) - cy;
        let ay = 1.0 - cy - by;
        Self { cx, bx, ax, cy, by, ay }
    }

    fn sample_x(&self, t: f64) -> f64 { ((self.ax * t + self.bx) * t + self.cx) * t }
    fn sample_y(&self, t: f64) -> f64 { ((self.ay * t + self.by) * t + self.cy) * t }
    fn sample_dx(&self, t: f64) -> f64 { (3.0 * self.ax * t + 2.0 * self.bx) * t + self.cx }

    fn solve_t(&self, x: f64) -> f64 {
        // Newton's method
        let mut t = x;
        for _ in 0..8 {
            let dx = self.sample_x(t) - x;
            if dx.abs() < 1e-7 { return t; }
            let d = self.sample_dx(t);
            if d.abs() < 1e-6 { break; }
            t -= dx / d;
        }
        // Bisection fallback
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        t = x;
        loop {
            let x2 = self.sample_x(t);
            if (x2 - x).abs() < 1e-7 { return t; }
            if x > x2 { lo = t; } else { hi = t; }
            t = (lo + hi) * 0.5;
            if hi - lo < 1e-7 { return t; }
        }
    }

    /// Given time progress x ∈ [0,1], returns eased value y
    fn solve(&self, x: f64) -> f64 { self.sample_y(self.solve_t(x)) }
}

// ──────────────────────────────────────────────────────────────────────────
// FlingCurve (classic, default desktop)
// ──────────────────────────────────────────────────────────────────────────
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
        let curve_duration = fc_time_at_vel(0.0);
        let max_start_speed = velocity.abs().min(fc_velocity(0.0));
        let time_offset = fc_time_at_vel(max_start_speed);
        let position_offset = fc_position(time_offset);
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

    fn tick(&mut self) -> (f64, bool) {
        let offset_time = self.start.elapsed().as_secs_f64() + self.time_offset;
        let (scalar, active) = if offset_time < self.curve_duration {
            self.velocity = fc_velocity(offset_time) * self.displacement_ratio;
            (fc_position(offset_time) - self.position_offset, true)
        } else {
            self.velocity = 0.0;
            (fc_position(self.curve_duration) - self.position_offset, false)
        };
        let total = scalar * self.displacement_ratio;
        let delta = total - self.cumulative_scroll;
        self.cumulative_scroll = total;
        (delta, active)
    }
}

// ──────────────────────────────────────────────────────────────────────────
// PhysicsBasedFlingCurve (physics_based_fling_curve.cc)
// Enabled via chrome://flags#experimental-fling-animation
//
// Key difference from FlingCurve:
//   - Distance derived from real physical deceleration (DPI-aware)
//   - Easing via cubic bezier (not raw exponential)
//   - Duration = min(2s, velocity / kDefaultPixelDeceleration)
//   - Bezier control points adjusted to match initial velocity slope
//   - Bounded by viewport size × boost_multiplier
// ──────────────────────────────────────────────────────────────────────────
struct PhysicsBasedFlingCurve {
    start: Instant,
    distance: f64,          // total scroll distance (signed px)
    curve_duration: f64,    // seconds
    bezier: CubicBezier,
    prev_offset: f64,
    prev_elapsed: f64,
    pub velocity: f64,
}

impl PhysicsBasedFlingCurve {
    fn new(velocity: f64, viewport_height: f64, boost_multiplier: f64) -> Self {
        // Convert velocity px/s → px/ms (Chromium: ScaleVector2d(velocity, 1/1000))
        let vel_ms = velocity / 1000.0;

        // Deceleration in px/ms² from physical constant × DPI
        let decel = PHYSICAL_DECELERATION * DEFAULT_PIXELS_PER_INCH; // px/ms²

        // Duration in ms, then convert to seconds
        let duration_ms = (vel_ms.abs() / decel).min(MAX_CURVE_DURATION * 1000.0);
        let duration_s = duration_ms / 1000.0;

        // Distance: d = v*t - 0.5*a*t²  (in px, signed)
        let dist_px = (vel_ms.abs() - decel * duration_ms * 0.5) * duration_ms;
        let distance = dist_px.copysign(velocity);

        // Clamp distance to viewport × boost bounds
        let max_dist = viewport_height * boost_multiplier * DEFAULT_BOUNDS_MULTIPLIER;
        let distance = distance.clamp(-max_dist, max_dist);

        // Duration for bezier: min(2s, fling_vel / kDefaultPixelDeceleration)
        // kDefaultPixelDeceleration is in px/s
        let curve_duration = (velocity.abs() / DEFAULT_PIXEL_DECELERATION)
            .min(MAX_CURVE_DURATION);

        // Configure cubic bezier control points to match initial velocity slope
        // slope = velocity * duration / distance  (tangent at t=0 must match vel)
        let slope = if distance.abs() > f64::EPSILON {
            (velocity.abs() * curve_duration / distance.abs()).abs()
        } else {
            1.0
        };

        let (p1x, p1y) = if slope * P1X < 1.0 {
            (P1X, P1X * slope) // scale p1y up
        } else {
            (P1Y / slope, P1Y) // move p1x left
        };

        let bezier = CubicBezier::new(p1x, p1y, P2X, P2Y);

        Self {
            start: Instant::now(),
            distance,
            curve_duration,
            bezier,
            prev_offset: 0.0,
            prev_elapsed: 0.0,
            velocity,
        }
    }

    fn tick(&mut self) -> (f64, bool) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let x = elapsed / self.curve_duration; // normalized time [0, 1]

        if x < 1.0 {
            let progress = self.bezier.solve(x);
            let offset = self.distance * progress;

            // velocity = Δoffset / Δtime  (mirrors GetVelocityAtTime)
            let dt = elapsed - self.prev_elapsed;
            if dt > 0.0 {
                self.velocity = (offset - self.prev_offset) / dt;
            }

            let delta = offset - self.prev_offset;
            self.prev_offset = offset;
            self.prev_elapsed = elapsed;
            (delta, true)
        } else {
            // Final frame — travel remaining distance
            let delta = self.distance - self.prev_offset;
            self.prev_offset = self.distance;
            self.velocity = 0.0;
            (delta, false)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unified fling handle
// ──────────────────────────────────────────────────────────────────────────
enum ActiveFling {
    Classic(FlingCurve),
    Physics(PhysicsBasedFlingCurve),
}

impl ActiveFling {
    fn tick(&mut self) -> (f64, bool) {
        match self {
            Self::Classic(c) => c.tick(),
            Self::Physics(p) => p.tick(),
        }
    }
    fn velocity(&self) -> f64 {
        match self {
            Self::Classic(c) => c.velocity,
            Self::Physics(p) => p.velocity,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// FlingBooster (fling_booster.cc)
// ──────────────────────────────────────────────────────────────────────────
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
        if self.previous_fling_starting_velocity == 0.0 { return; }
        self.cutoff_time_for_boost = Some(Instant::now() + FLING_BOOST_TIMEOUT);
    }

    fn observe_scroll_update(&mut self, delta: f64) {
        if self.previous_fling_starting_velocity == 0.0 { return; }
        if self.cutoff_time_for_boost.is_some_and(|t| Instant::now() > t) { self.reset(); return; }
        if self.cutoff_time_for_boost.is_none() { return; }
        if (delta >= 0.0) != (self.previous_fling_starting_velocity >= 0.0) { self.reset(); return; }
        if let Some(prev_t) = self.previous_boosting_scroll_timestamp {
            let dt = prev_t.elapsed().as_secs_f64();
            if dt >= 0.001 {
                let sv = delta / dt;
                if sv * sv < MIN_BOOST_TOUCH_SCROLL_SPEED_SQ { self.reset(); return; }
            }
        }
        self.previous_boosting_scroll_timestamp = Some(Instant::now());
        self.cutoff_time_for_boost = Some(Instant::now() + FLING_BOOST_TIMEOUT);
    }

    fn observe_scroll_end(&mut self) { self.previous_boosting_scroll_timestamp = None; }

    fn observe_fling_cancel(&mut self, prevent_boosting: bool) {
        if prevent_boosting { self.reset(); return; }
        self.previous_boosting_scroll_timestamp = None;
        self.cutoff_time_for_boost = Some(Instant::now() + FLING_BOOST_TIMEOUT);
    }

    fn observe_progress_fling(&mut self, current_velocity: f64) {
        if self.previous_fling_starting_velocity == 0.0 { return; }
        self.current_fling_velocity = current_velocity;
    }

    fn should_boost_fling(&self, new_velocity: f64) -> bool {
        if self.previous_fling_starting_velocity == 0.0 { return false; }
        if self.cutoff_time_for_boost.is_none() { return false; }
        if self.cutoff_time_for_boost.is_some_and(|t| Instant::now() > t) { return false; }
        if (new_velocity >= 0.0) != (self.previous_fling_starting_velocity >= 0.0) { return false; }
        if self.current_fling_velocity * self.current_fling_velocity < MIN_BOOST_FLING_SPEED_SQ { return false; }
        if new_velocity * new_velocity < MIN_BOOST_FLING_SPEED_SQ { return false; }
        true
    }

    pub fn reset(&mut self) {
        self.cutoff_time_for_boost = None;
        self.previous_fling_starting_velocity = 0.0;
        self.current_fling_velocity = 0.0;
        self.previous_boosting_scroll_timestamp = None;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// LSQ2 velocity tracker (velocity_tracker.cc, degree=2, WEIGHTING_NONE)
// ──────────────────────────────────────────────────────────────────────────
fn solve_least_squares(x: &[f64], y: &[f64], w: &[f64], m: usize, n: usize) -> Option<Vec<f64>> {
    let mut a = vec![vec![0.0f64; m]; n];
    for h in 0..m {
        a[0][h] = w[h];
        for i in 1..n { a[i][h] = a[i-1][h] * x[h]; }
    }
    let mut q = vec![vec![0.0f64; m]; n];
    let mut r = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        q[j] = a[j].clone();
        for i in 0..j {
            let dot: f64 = (0..m).map(|h| q[j][h] * q[i][h]).sum();
            for h in 0..m { q[j][h] -= dot * q[i][h]; }
        }
        let norm: f64 = (0..m).map(|h| q[j][h] * q[j][h]).sum::<f64>().sqrt();
        if norm < 1e-6 { return None; }
        let inv = 1.0 / norm;
        for h in 0..m { q[j][h] *= inv; }
        for i in 0..n {
            r[j][i] = if i < j { 0.0 } else { (0..m).map(|h| q[j][h] * a[i][h]).sum() };
        }
    }
    let wy: Vec<f64> = (0..m).map(|h| y[h] * w[h]).collect();
    let mut b = vec![0.0f64; n];
    for i in (0..n).rev() {
        b[i] = (0..m).map(|h| q[i][h] * wy[h]).sum();
        for j in (i+1)..n { b[i] -= r[i][j] * b[j]; }
        b[i] /= r[i][i];
    }
    Some(b)
}

struct LsqVelocityTracker {
    samples: VecDeque<(Instant, f64)>,
    cumulative: f64,
}

impl LsqVelocityTracker {
    fn new() -> Self { Self { samples: VecDeque::new(), cumulative: 0.0 } }

    fn push(&mut self, delta: f64) {
        self.cumulative += delta;
        let now = Instant::now();
        self.samples.retain(|(t, _)| now.duration_since(*t).as_secs_f64() * 1000.0 < HORIZON_MS);
        self.samples.push_back((now, self.cumulative));
        while self.samples.len() > HISTORY_SIZE { self.samples.pop_front(); }
    }

    fn velocity(&self) -> f64 {
        let m = self.samples.len();
        if m < 2 { return 0.0; }
        let newest_t = self.samples.back().unwrap().0;
        let times: Vec<f64> = self.samples.iter()
            .map(|(t, _)| -(newest_t.duration_since(*t).as_secs_f64())).collect();
        let positions: Vec<f64> = self.samples.iter().map(|(_, p)| *p).collect();
        let weights = vec![1.0f64; m];
        let degree = 2.min(m - 1);
        match solve_least_squares(&times, &positions, &weights, m, degree + 1) {
            Some(b) if b.len() >= 2 => b[1],
            _ => 0.0,
        }
    }

    fn clear(&mut self) { self.samples.clear(); self.cumulative = 0.0; }
}

fn wheel_detent_step(page_size: f64) -> f64 { page_size.powf(2.0 / 3.0) }

// ──────────────────────────────────────────────────────────────────────────
// App
// ──────────────────────────────────────────────────────────────────────────
const ITEM_H: f32 = 60.0;
const ITEM_N: usize = 500;
const CONTENT_H: f64 = ITEM_H as f64 * ITEM_N as f64;

struct App {
    offset: f64,
    fling: Option<ActiveFling>,
    booster: FlingBooster,
    tracker: LsqVelocityTracker,
    last_scroll: Option<Instant>,
    scroll_began: bool,
    is_trackpad: bool,
    use_physics_curve: bool, // Space to toggle
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
            use_physics_curve: false,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        // Space toggles between FlingCurve and PhysicsBasedFlingCurve
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.use_physics_curve = !self.use_physics_curve;
            self.fling = None;
            self.booster.reset();
        }

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
                if self.fling.is_some() {
                    self.booster.observe_fling_cancel(false);
                    self.fling = None;
                }
                if !self.scroll_began {
                    self.booster.observe_scroll_begin();
                    self.scroll_began = true;
                }
                let scaled = if is_trackpad {
                    -raw_delta as f64 * MAGIC_SCROLL_FACTOR
                } else {
                    -raw_delta as f64 / 50.0 * step
                };
                self.booster.observe_scroll_update(scaled);
                self.tracker.push(scaled);
                self.last_scroll = Some(Instant::now());
                self.offset = (self.offset + scaled).clamp(0.0, max_offset);
                ctx.request_repaint();
            } else if self.last_scroll.is_some_and(|t| t.elapsed().as_millis() > 30) {
                self.scroll_began = false;
                let raw_vel = self.tracker.velocity();
                if raw_vel.abs() > 20.0 {
                    self.booster.observe_scroll_end();
                    let vel = self.booster.get_velocity_for_fling_start(raw_vel);
                    self.fling = Some(if self.use_physics_curve {
                        ActiveFling::Physics(PhysicsBasedFlingCurve::new(
                            vel, view_h, 1.0,
                        ))
                    } else {
                        ActiveFling::Classic(FlingCurve::new(vel))
                    });
                } else {
                    self.booster.reset();
                }
                self.last_scroll = None;
                self.tracker.clear();
            }

            if let Some(f) = &mut self.fling {
                let (delta, running) = f.tick();
                self.booster.observe_progress_fling(f.velocity());
                if delta.abs() > 0.1 {
                    self.offset = (self.offset + delta).clamp(0.0, max_offset);
                }
                if !running {
                    self.booster.reset();
                    self.fling = None;
                }
                ctx.request_repaint();
            }

            // ── draw items ──────────────────────────────────────────────────
            let painter = ui.painter_at(rect);
            ui.set_clip_rect(rect);

            for i in 0..ITEM_N {
                let y = i as f32 * ITEM_H - self.offset as f32;
                if y + ITEM_H < 0.0 || y > rect.height() { continue; }
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
                let thumb_y = (self.offset / max_offset) as f32 * (rect.height() - thumb_h);
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(rect.right() - 6.0, rect.top() + thumb_y),
                        egui::vec2(4.0, thumb_h),
                    ),
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(160, 160, 160, 200),
                );
            }

            // debug bar
            let curve_name = if self.use_physics_curve {
                "PhysicsBasedFlingCurve [SPACE to switch]"
            } else {
                "FlingCurve (default) [SPACE to switch]"
            };
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(format!(
                    "curve: {}  offset: {:.0}  vel: {:.0}  boost: {:.0}  input: {}",
                    curve_name,
                    self.offset,
                    self.fling.as_ref().map_or(0.0, |f| f.velocity()),
                    self.booster.previous_fling_starting_velocity,
                    if self.is_trackpad { "trackpad" } else { "wheel" },
                ));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Chromium Scroll Replica",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}