
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}

impl Vec3 {
    fn length_squared(self) -> f64 {
        self.x*self.x + self.y*self.y + self.z*self.z
    }
    fn length(self) -> f64 { self.length_squared().sqrt() }
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
    fn scale(self, n: f64) -> Vec3 {
        Vec3 { x: self.x * n, y: self.y * n, z: self.z * n }
    }
    fn unit(self) -> Vec3 { self.scale(1.0 / self.length()) }
    fn neg(self) -> Vec3 {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }
    fn dot(self, other: Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    fn cross(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.y*other.z - self.z*other.y,
            y: self.z*other.x - self.x*other.z,
            z: self.x*other.y - self.y*other.x
        }
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Ray {
    origin: Vec3,
    dir: Vec3
}

impl Ray {
    fn at(self, t: f64) -> Vec3 { self.origin.add(self.dir.scale(t)) }
}

fn ray_color(r: &Ray) -> Vec3 {
    let unit_dir = r.dir.unit();
    let t = (unit_dir.y + 1.0)*0.5;
    return (Vec3 { x: 1.0, y: 1.0, z: 1.0 }).scale(1.0 - t)
        .add((Vec3 { x: 0.5, y: 0.7, z: 1.0 }).scale(t));
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height = ((image_width as f64) / aspect_ratio) as i32;

    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;

    let origin = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    let horizontal = Vec3 { x: viewport_width, y: 0.0, z: 0.0 };
    let vertical = Vec3 { x: 0.0, y: viewport_height, z: 0.0 };
    let lower_left_corner = origin.add(horizontal.scale(0.5).neg())
        .add(vertical.scale(0.5).neg())
        .add(Vec3 { x: 0.0, y: 0.0, z: -focal_length });

    println!("P3\n{} {}\n255", image_width, image_height);
    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let u = (i as f64) / ((image_width - 1) as f64);
            let v = (j as f64) / ((image_height - 1) as f64);
            let r = Ray {
                origin: origin,
                dir: lower_left_corner.add(horizontal.scale(u))
                    .add(vertical.scale(v)).add(origin.neg())
            };
            let color = ray_color(&r).scale(255.0);
            println!("{} {} {}", color.x as i32, color.y as i32, color.z as i32);
        }
    }
}
