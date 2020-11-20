
use std::fmt;
use std::vec::Vec;
use std::boxed::Box;
use std::rc::Rc;
use rand::Rng;

#[derive(Copy, Clone)]
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
    fn pointwise_mult(self, other: Vec3) -> Vec3 {
        Vec3 { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z }
    }
    fn near_zero(self) -> bool {
        let s = 1e-8;
        return self.x.abs() < s && self.y.abs() < s && self.z.abs() < s;
    }
    fn reflect(self, n: Vec3) -> Vec3 {
        self.add(n.scale(self.dot(n) * 2.0).neg())
    }
}
impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.x, self.y, self.z)
    }
}

#[derive(Copy, Clone)]
struct Ray {
    origin: Vec3,
    dir: Vec3
}
impl Ray {
    fn at(self, t: f64) -> Vec3 { self.origin.add(self.dir.scale(t)) }
}

#[derive(Copy, Clone)]
struct HitRecord<'a> {
    point: Vec3, normal: Vec3, t: f64, front_face: bool, mat: &'a dyn Material
}
impl HitRecord<'_> {
    fn set_face_normal(&mut self, r: Ray, outward_normal: Vec3) {
        self.front_face = r.dir.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            outward_normal.neg()
        };
    }
}

trait Material {
    fn scatter(
        &self, r_in: Ray, rec: &HitRecord, attenuation: &mut Vec3, scattered: &mut Ray
    ) -> bool;
}

#[derive(Copy, Clone)]
struct Metal {
    albedo: Vec3,
    fuzz: f64
}
impl Material for Metal {
    fn scatter(
        &self, r_in: Ray, rec: &HitRecord, attenuation: &mut Vec3, scattered: &mut Ray
    ) -> bool {
        let reflected = r_in.dir.unit().reflect(rec.normal);
        *scattered = Ray {
            origin: rec.point,
            dir: reflected.add(random_unit_vec3(false).scale(self.fuzz))
        };
        *attenuation = self.albedo;
        return scattered.dir.dot(rec.normal) > 0.0;
    }
}

#[derive(Copy, Clone)]
struct Lambertian {
    albedo: Vec3,
}
impl Material for Lambertian {
    fn scatter(
        &self, r_in: Ray, rec: &HitRecord, attenuation: &mut Vec3, scattered: &mut Ray
    ) -> bool {
        let mut scatter_direction = rec.normal.add(random_unit_vec3(false));
        if scatter_direction.near_zero() {
            scatter_direction = rec.normal;
        }
        *scattered = Ray { origin: rec.point, dir: scatter_direction };
        *attenuation = self.albedo;
        return true;
    }
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = uv.neg().dot(n).min(1.0);
    let r_out_perp = uv.add(n.scale(cos_theta)).scale(etai_over_etat);
    let r_out_parallel = n.scale(-((1.0 - r_out_perp.length_squared()).abs().sqrt()));
    return r_out_perp.add(r_out_parallel);
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    let mut r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 *= r0;
    return r0 + (1.0 - r0)*(1.0 - cosine).powi(5);
}

struct Dielectric { ir: f64 }
impl Material for Dielectric {
    fn scatter(
        &self, r_in: Ray, rec: &HitRecord, attenuation: &mut Vec3, scattered: &mut Ray
    ) -> bool {
        *attenuation = Vec3 { x: 1.0, y: 1.0, z: 1.0 };
        let refraction_ratio = if rec.front_face { 1.0 / self.ir } else { self.ir };

        let unit_dir = r_in.dir.unit();
        let cos_theta = unit_dir.neg().dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;

        let mut rng = rand::thread_rng();
        let rand_f64: f64 = rng.gen();
        let dir =
            if cannot_refract || reflectance(cos_theta, refraction_ratio) > rand_f64 {
                unit_dir.reflect(rec.normal)
            } else {
                refract(unit_dir, rec.normal, refraction_ratio)
            };

        *scattered = Ray { origin: rec.point, dir: dir };
        return true;
    }
}

trait Hittable {
    fn hit(&self, r: Ray, t_min: f64, t_max: f64) -> Option<HitRecord>;
}

#[derive(Copy, Clone)]
struct Sphere<'a> { center: Vec3, radius: f64, mat: &'a dyn Material }
impl Hittable for Sphere<'_> {
    fn hit(&self, r: Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let oc = r.origin.add(self.center.neg());
        let a = r.dir.length_squared();
        let half_b = oc.dot(r.dir);
        let c = oc.length_squared() - self.radius*self.radius;
        let discriminant = half_b*half_b - a*c;
        if discriminant < 0.0 { return None; }

        let root = discriminant.sqrt();
        let mut tmp = (-half_b - root) / a;
        if tmp >= t_max || tmp <= t_min { tmp = (-half_b + root) / a; }
        if tmp >= t_max || tmp <= t_min { return None; }
        let pt = r.at(tmp);
        let norm = pt.add(self.center.neg()).scale(1.0 / self.radius);
        let mut rec = HitRecord {
            t: tmp, point: pt, normal: norm, front_face: false, mat: self.mat
        };
        rec.set_face_normal(r, norm);
        return Some(rec);
    }
}

struct HittableList { objects: Vec<Box<dyn Hittable>> }
impl Hittable for HittableList {
    fn hit(&self, r: Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut rec: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for obj in self.objects.iter() {
            if let Some(tmp_rec) = obj.hit(r, t_min, closest_so_far) {
                closest_so_far = tmp_rec.t;
                rec = Some(tmp_rec.clone());
            }
        }

        return rec;
    }
}

#[derive(Copy, Clone)]
struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f64
}
impl Camera {
    fn get_ray(&self, s: f64, t: f64) -> Ray {
        let rd = random_unit_vec3(true).scale(self.lens_radius);
        let offset = self.u.scale(rd.x).add(self.v.scale(rd.y));

        Ray {
            origin: self.origin.add(offset),
            dir: self.lower_left_corner.add(self.horizontal.scale(t))
                .add(self.vertical.scale(t)).add(self.origin.neg())
                .add(offset.neg())
        }
    }
}
fn init_camera(
    lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: f64,
    aspect_ratio: f64, aperture: f64, focus_dist: f64
) -> Camera {
    let theta = vfov.to_radians();
    let h = (theta / 2.0).tan();
    let viewport_height = 2.0 * h;
    let viewport_width = aspect_ratio * viewport_height;

    let w = lookfrom.add(lookat.neg()).unit();
    let u = vup.cross(w).unit();
    let v = w.cross(u);

    let origin = lookfrom;
    let horizontal = u.scale(viewport_width * focus_dist);
    let vertical = v.scale(viewport_height * focus_dist);
    let lower_left_corner = origin.add(horizontal.scale(0.5).neg())
        .add(vertical.scale(0.5).neg())
        .add(w.scale(focus_dist).neg());

    return Camera {
        origin: origin, lower_left_corner: lower_left_corner,
        horizontal: horizontal, vertical: vertical,
        u: u, v: v, lens_radius: aperture / 2.0
    };
}

fn clamp(i: f64, l: f64, h: f64) -> f64 {
    if i < l { l } else if i > h { h } else { i }
}

fn write_color(color: Vec3, samples_per_pixel: i32) {
    let mut r = color.x;
    let mut g = color.y;
    let mut b = color.z;
    let scale = 1.0 / (samples_per_pixel as f64);
    r = (r * scale).sqrt();
    g = (g * scale).sqrt();
    b = (b * scale).sqrt();
    println!("{} {} {}",
             (256.0 * clamp(r, 0., 0.999)) as i32,
             (256.0 * clamp(g, 0., 0.999)) as i32,
             (256.0 * clamp(b, 0., 0.999)) as i32
    );
}

fn random_unit_vec3(in_unit_disk: bool) -> Vec3 {
    let mut rng = rand::thread_rng();
    loop {
        let x: f64 = rng.gen();
        let y: f64 = rng.gen();
        let z: f64 = if in_unit_disk { 0.5 } else { rng.gen() };
        let p = Vec3 { x: x * 2.0 - 1.0, y: y * 2.0 - 1.0, z: z * 2.0 - 1.0 };
        if p.length_squared() >= 1.0 { continue; }
        return p.unit();
    }
}

fn ray_color(r: Ray, world: &dyn Hittable, depth: i32) -> Vec3 {
    if depth <= 0 { return Vec3 { x: 0.0, y: 0.0, z: 0.0 }; }

    if let Some(rec) = world.hit(r, 0.001, std::f64::INFINITY) {
        let mut scattered = r;
        let mut attenuation = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
        if rec.mat.scatter(r, &rec, &mut attenuation, &mut scattered) {
            return attenuation.pointwise_mult(ray_color(scattered, world, depth-1));
        }
        return Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    }
    let unit_dir = r.dir.unit();
    let t = (unit_dir.y + 1.0)*0.5;
    return (Vec3 { x: 1.0, y: 1.0, z: 1.0 }).scale(1.0 - t)
        .add((Vec3 { x: 0.5, y: 0.7, z: 1.0 }).scale(t));
}

fn random_scene() -> HittableList {
    let mut rng = rand::thread_rng();
    let mut world = HittableList { objects: Vec::new() };
    for i in -11..11 {
        for j in -11..11 {
            let choose_mat: f64 = rng.gen();
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();
            let center = Vec3 { x: 0.9 * r1, y: 0.2, z: (j as f64) + 0.9 * r2 };

            if center.add(Vec3 { x: -4., y: -0.2, z: 0. }).length() > 0.9 {
                // TODO
            }
        }
    }

    return world;
}

fn main() {
    let aspect_ratio = 3.0 / 2.0;
    let image_width = 1200;
    let image_height = ((image_width as f64) / aspect_ratio) as i32;
    let samples_per_pixel = 500;
    let max_depth = 50;

    let world = random_scene();
    let lookfrom = Vec3 { x: 13., y: 2., z: 3. };
    let lookat = Vec3 { x: 0., y: 0., z: 0. };
    let vup = Vec3 { x: 0., y: 1., z: 0. };
    let dist_to_focus = 10.0;
    let aperture = 0.1;
    let cam = init_camera(
        lookfrom, lookat, vup, 20.0,
        aspect_ratio, aperture, dist_to_focus
    );

    let mut rng = rand::thread_rng();

    println!("P3\n{} {}\n255", image_width, image_height);
    for j in (0..image_height).rev() {
        for i in 0..image_width {
            let mut color = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
            for _ in 0..samples_per_pixel {
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();
                let u = (i as f64 + r1) / ((image_width - 1) as f64);
                let v = (j as f64 + r2) / ((image_height - 1) as f64);
                let r = cam.get_ray(u, v);
                color = color.add(ray_color(r, &world, max_depth));
            }
            write_color(color, samples_per_pixel);
        }
    }
}
