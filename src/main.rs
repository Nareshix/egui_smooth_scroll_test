use eframe::egui;
use std::collections::VecDeque;
use std::time::Instant;

const DECEL_FRICTION: f64 = 4.0;
const OVERSHOOT_FRICTION: f64 = 20.0;
const MAX_OVERSHOOT: f64 = 100.0;
const MAGIC_SCROLL_FACTOR: f64 = 2.5;
const VELOCITY_ACCUMULATION_FLOOR: f64 = 0.33;
const VELOCITY_ACCUMULATION_CEIL: f64 = 1.0;
const VELOCITY_ACCUMULATION_MAX: f64 = 6.0;

#[derive(PartialEq)]
enum Phase {
    Decelerating,
    Overshooting,
    Finished,
}

struct KineticScrolling {
    phase: Phase,
    lower: f64,
    upper: f64,
    c1: f64,
    c2: f64,
    equilibrium: f64,
    t: Instant,
    pub position: f64,
    pub velocity: f64,
}

impl KineticScrolling {
    fn new(lower: f64, upper: f64, pos: f64, vel: f64) -> Self {
        let mut s = Self {
            phase: Phase::Decelerating,
            lower,
            upper,
            c1: vel / DECEL_FRICTION + pos,
            c2: -vel / DECEL_FRICTION,
            equilibrium: 0.0,
            t: Instant::now(),
            position: pos,
            velocity: vel,
        };
        if pos < lower {
            s.init_overshoot(lower, pos, vel);
        } else if pos > upper {
            s.init_overshoot(upper, pos, vel);
        }
        s
    }

    fn init_overshoot(&mut self, eq: f64, pos: f64, vel: f64) {
        self.phase = Phase::Overshooting;
        self.equilibrium = eq;
        self.c1 = pos - eq;
        self.c2 = vel + OVERSHOOT_FRICTION / 2.0 * self.c1;
        self.t = Instant::now();
    }

    fn tick(&mut self) -> (f64, bool) {
        let t = self.t.elapsed().as_secs_f64();

        match self.phase {
            Phase::Decelerating => {
                let e = (-DECEL_FRICTION * t).exp();
                self.position = self.c1 + self.c2 * e;
                self.velocity = -DECEL_FRICTION * self.c2 * e;

                if self.position < self.lower {
                    self.init_overshoot(self.lower, self.position, self.velocity);
                } else if self.position > self.upper {
                    self.init_overshoot(self.upper, self.position, self.velocity);
                } else if self.velocity.abs() < 0.1 {
                    self.phase = Phase::Finished;
                    self.position = self.position.round();
                }
            }
            Phase::Overshooting => {
                let half = MAX_OVERSHOOT / 2.0;
                let e = (-OVERSHOOT_FRICTION / 2.0 * t).exp();
                let mut pos = e * (self.c1 + self.c2 * t);

                if pos < self.lower - half || pos > self.upper + half {
                    pos = pos.clamp(self.lower - half, self.upper + half);
                    self.init_overshoot(self.equilibrium, pos, 0.0);
                } else {
                    self.velocity = self.c2 * e - OVERSHOOT_FRICTION / 2.0 * pos;
                }

                self.position = pos + self.equilibrium;

                if pos.abs() < 0.1 {
                    self.phase = Phase::Finished;
                    self.position = self.equilibrium;
                    self.velocity = 0.0;
                }
            }
            Phase::Finished => {}
        }

        (self.position, self.phase != Phase::Finished)
    }

    fn stop(&mut self) {
        if self.phase == Phase::Decelerating {
            self.phase = Phase::Finished;
            self.position = self.position.round();
        }
    }
}

fn accumulate_velocity(kinetic: &mut Option<KineticScrolling>, velocity: &mut f64) {
    let Some(k) = kinetic else { return };

    let last_velocity = k.velocity;
    let same_direction = (*velocity >= 0.0) == (last_velocity >= 0.0);
    let above_floor = velocity.abs() >= last_velocity.abs() * VELOCITY_ACCUMULATION_FLOOR;

    if same_direction && above_floor {
        let min_vel = last_velocity * VELOCITY_ACCUMULATION_FLOOR;
        let max_vel = last_velocity * VELOCITY_ACCUMULATION_CEIL;
        let range = max_vel - min_vel;
        if range.abs() > f64::EPSILON {
            let mult = (*velocity - min_vel) / range;
            *velocity += last_velocity * mult.min(VELOCITY_ACCUMULATION_MAX);
        }
    }
}

fn wheel_detent_step(page_size: f64) -> f64 {
    page_size.powf(2.0 / 3.0)
}

struct VelocityTracker {
    history: VecDeque<(Instant, f64)>,
}

impl VelocityTracker {
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
        }
    }

    fn push(&mut self, delta: f64) {
        let now = Instant::now();
        self.history
            .retain(|(t, _)| now.duration_since(*t).as_millis() < 100);
        self.history.push_back((now, delta));
    }

    fn velocity(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let total: f64 = self.history.iter().map(|(_, d)| d).sum();
        let span = self.history.front().unwrap().0.elapsed().as_secs_f64();
        if span > 0.0 {
            total / span
        } else {
            0.0
        }
    }

    fn clear(&mut self) {
        self.history.clear();
    }
}

const ITEM_H: f32 = 60.0;
const ITEM_N: usize = 50;
const CONTENT_H: f64 = ITEM_H as f64 * ITEM_N as f64;

struct App {
    offset: f64,
    kinetic: Option<KineticScrolling>,
    tracker: VelocityTracker,
    last_scroll: Option<Instant>,
    is_trackpad: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            offset: 0.0,
            kinetic: None,
            tracker: VelocityTracker::new(),
            last_scroll: None,
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
                if let Some(k) = &mut self.kinetic {
                    k.stop();
                }

                let scaled = if is_trackpad {
                    -raw_delta as f64 * MAGIC_SCROLL_FACTOR
                } else {
                    -raw_delta as f64 / 50.0 * step
                };

                self.tracker.push(scaled);
                self.last_scroll = Some(Instant::now());
                self.offset = (self.offset + scaled).clamp(0.0, max_offset);
                ctx.request_repaint();
            } else if self
                .last_scroll
                .is_some_and(|t| t.elapsed().as_millis() > 30)
            {
                let mut vel = self.tracker.velocity();

                if vel.abs() > 20.0 {
                    accumulate_velocity(&mut self.kinetic, &mut vel);
                    self.kinetic = Some(KineticScrolling::new(0.0, max_offset, self.offset, vel));
                }

                self.last_scroll = None;
                self.tracker.clear();
            }

            if let Some(k) = &mut self.kinetic {
                let (pos, running) = k.tick();
                self.offset = pos.clamp(0.0, max_offset);
                if running {
                    ctx.request_repaint();
                } else {
                    self.kinetic = None;
                }
            }

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
                    format!("Item {:02}", i + 1),
                    egui::FontId::proportional(16.0),
                    egui::Color32::WHITE,
                );
            }

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

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.label(format!(
                    "offset: {:.0}  kinetic_vel: {:.0}  input: {}  step: {:.1}",
                    self.offset,
                    self.kinetic.as_ref().map_or(0.0, |k| k.velocity),
                    if self.is_trackpad {
                        "trackpad"
                    } else {
                        "wheel"
                    },
                    step,
                ));
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Kinetic Scroll",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}
