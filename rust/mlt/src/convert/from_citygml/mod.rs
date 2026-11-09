//! `mlt convert <citygml> <dir>`: tile the buildings of `CityGML` files into an MLT `z/x/y.mlt`
//! tree of v2 `Meshes` layers (XYZ scheme, `y = 0` at the north).
//!
//! Each building, bridge or tunnel becomes one mesh at the highest LOD of each of its parts,
//! reprojected from its file's `srsName`. As `docs/specification/v2.md` asks, a mesh is stored
//! whole in the tile of each zoom that holds the centre of its bounding box, and its z values are
//! heights above its lowest vertex. Its vertices are welded on the tile's grid, and each triangle
//! carries the `part` m-value of the semantic surface it lies in: wall, roof, ground, window or
//! door. A surface without semantics is a roof where it faces up and a wall otherwise.

mod crs;
mod parser;
mod triangulate;

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result as AnyResult};
use mlt_core::geo_types::{Geometry, MultiPoint, Point};
use mlt_core::{MValue, MValueScope, PropKind, PropValue, TileLayer, ZStep};
use rayon::prelude::*;
use walkdir::WalkDir;

use self::crs::{Reprojector, epsg_of_srs_name};
use self::parser::Part;
use super::{ConvertArgs, Reencoder};

/// The z grid without `--z-step`: 1 cm.
const DEFAULT_Z_STEP: i8 = -2;
/// The finest grid unit the extent of a tile aims for, in metres.
const UNIT_METRES: f64 = 0.01;
/// The largest extent a v2 layer header can name.
const MAX_EXTENT: u32 = 1 << 21;
/// The WGS84 semi-major axis in metres.
const WGS84_A: f64 = 6_378_137.0;
/// The up component of a unit normal above which a surface without semantics is a roof, about
/// 70 degrees from vertical.
const ROOF_MIN_UP: f64 = 0.34;

/// Whether `path` is a `CityGML` file, or a directory whose first `.gml` file comes before any tile.
pub(super) fn is_citygml(path: &Path) -> bool {
    if path.is_file() {
        return is_gml(path);
    }
    WalkDir::new(path)
        .sort_by_file_name()
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .find_map(|e| match e.path().extension().and_then(|x| x.to_str()) {
            Some("mlt" | "mvt" | "pbf") => Some(false),
            _ if is_gml(e.path()) => Some(true),
            _ => None,
        })
        .unwrap_or(false)
}

fn is_gml(path: &Path) -> bool {
    path.extension()
        .is_some_and(|x| x.eq_ignore_ascii_case("gml"))
}

/// A surface of one part, with geodetic positions: longitude and latitude in radians, and metres up.
struct Surface {
    part: Part,
    positions: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
}

struct Building {
    gml_id: Option<String>,
    surfaces: Vec<Surface>,
}

/// Tile the `CityGML` file or directory `args.input` into the directory `args.output`.
/// [`super::convert`] has already checked the flags and that the output does not exist.
pub(super) fn convert(args: &ConvertArgs, reencoder: &Reencoder) -> AnyResult<()> {
    let (input, output) = (args.input.as_path(), args.output.as_path());
    let (min_zoom, max_zoom) = args.tiled_zooms()?;
    let name = args.tiled_layer_name()?;
    let z_step = ZStep::above_ground(args.z_step.unwrap_or(DEFAULT_Z_STEP))?;

    let files = files(input)?;
    let read: Vec<Vec<Building>> = files
        .par_iter()
        .map(|path| read(path).with_context(|| format!("reading {}", path.display())))
        .collect::<AnyResult<_>>()?;
    let buildings: Vec<Building> = read.into_iter().flatten().collect();

    let mut tiles: BTreeMap<Tile, Vec<(u64, &Building, Mesh)>> = BTreeMap::new();
    for (id, building) in (0..).zip(&buildings) {
        for (tile, mesh) in place(building, min_zoom..=max_zoom, z_step) {
            tiles.entry(tile).or_default().push((id, building, mesh));
        }
    }
    let written = tiles.len();
    tiles
        .into_par_iter()
        .try_for_each(|(tile, meshes)| -> AnyResult<()> {
            let at = format!("{}/{}/{}", tile.zoom, tile.x, tile.y);
            let layer = layer(&name, tile, z_step, meshes)?;
            let bytes = reencoder
                .encode(layer)
                .with_context(|| format!("encoding tile {at}"))?;
            let dir = output.join(tile.zoom.to_string()).join(tile.x.to_string());
            fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            let path = dir.join(format!("{}.mlt", tile.y));
            fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))
        })?;
    eprintln!(
        "{} -> {}: wrote {written} tile(s) holding {} building(s) of {} file(s) at zooms {min_zoom}..={max_zoom}",
        input.display(),
        output.display(),
        buildings.len(),
        files.len(),
    );
    Ok(())
}

/// The `CityGML` file `input`, or the `.gml` files under the directory `input`, sorted.
fn files(input: &Path) -> AnyResult<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_owned()]);
    }
    let mut files = Vec::new();
    for entry in WalkDir::new(input).sort_by_file_name() {
        let entry = entry.with_context(|| format!("searching {}", input.display()))?;
        if entry.file_type().is_file() && is_gml(entry.path()) {
            files.push(entry.into_path());
        }
    }
    Ok(files)
}

/// The buildings of the `CityGML` file at `path`, triangulated and reprojected.
fn read(path: &Path) -> AnyResult<Vec<Building>> {
    let bytes = fs::read(path)?;
    let document = parser::parse(&bytes)?;
    if document.buildings.is_empty() {
        return Ok(Vec::new());
    }
    let srs_name = document
        .srs_name
        .context("the file names no CRS in an srsName")?;
    let epsg = epsg_of_srs_name(&srs_name)
        .with_context(|| format!("the srsName {srs_name:?} names no EPSG code"))?;
    let reprojector = Reprojector::new(epsg)?;
    let mut buildings = Vec::with_capacity(document.buildings.len());
    for building in document.buildings {
        let mut surfaces = Vec::with_capacity(building.polygons.len());
        for polygon in building.polygons {
            let mut triangles = Vec::new();
            let Some(normal) = triangulate::polygon(
                &polygon.points,
                &polygon.ends,
                reprojector.geographic,
                &mut triangles,
            ) else {
                continue;
            };
            let part = polygon.part.unwrap_or_else(|| {
                if normal[2] / triangulate::dot(normal, normal).sqrt() > ROOF_MIN_UP {
                    Part::Roof
                } else {
                    Part::Wall
                }
            });
            let mut positions = polygon.points;
            reprojector.to_geodetic(&mut positions)?;
            surfaces.push(Surface {
                part,
                positions,
                triangles,
            });
        }
        if !surfaces.is_empty() {
            buildings.push(Building {
                gml_id: building.gml_id,
                surfaces,
            });
        }
    }
    Ok(buildings)
}

/// A tile of the XYZ scheme, `y = 0` at the north.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Tile {
    zoom: u8,
    x: u32,
    y: u32,
}

impl Tile {
    /// The tile of `zoom` holding the Web Mercator position `[x, y]` of [`unit()`].
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the unit coordinates are clamped to [0, 1), so the products fit below 2^zoom"
    )]
    fn of(unit: [f64; 2], zoom: u8) -> Self {
        let n = f64::from(1_u32 << zoom);
        let [x, y] = unit.map(|u| (u.clamp(0.0, 1.0 - f64::EPSILON) * n) as u32);
        Self { zoom, x, y }
    }

    /// The extent whose unit is between 1 and 2 cm across the middle of the tile.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the exponent is clamped to 6..=21"
    )]
    fn extent(self) -> u32 {
        let n = f64::from(1_u32 << self.zoom);
        let lat = (PI * (1.0 - 2.0 * (f64::from(self.y) + 0.5) / n))
            .sinh()
            .atan();
        let width = 2.0 * PI * WGS84_A * lat.cos() / n;
        let exponent = (width / UNIT_METRES).log2().floor().clamp(6.0, 21.0) as u32;
        (1 << exponent).min(MAX_EXTENT)
    }

    /// `[x, y]` on this tile's grid of `extent` of the Web Mercator position `[x, y]`, rounded
    /// half to even, or `None` if it does not fit an `i32`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rounded values are checked to fit"
    )]
    fn grid(self, [ux, uy]: [f64; 2], extent: u32) -> Option<[i32; 2]> {
        let n = f64::from(1_u32 << self.zoom);
        let extent = f64::from(extent);
        let x = ((ux * n - f64::from(self.x)) * extent).round_ties_even();
        let y = ((uy * n - f64::from(self.y)) * extent).round_ties_even();
        let fits = |v: f64| (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&v);
        (fits(x) && fits(y)).then_some([x as i32, y as i32])
    }
}

/// Web Mercator `[x, y]` of `[lon, lat]` in radians, the world spanning `[0, 1]`.
fn unit(lon: f64, lat: f64) -> [f64; 2] {
    [
        (lon + PI) / (2.0 * PI),
        (1.0 - lat.tan().asinh() / PI) / 2.0,
    ]
}

/// A building on a tile's grid: its vertices, the triangles between them that still have three
/// distinct corners, and the part of each triangle.
#[derive(Debug, PartialEq)]
struct Mesh {
    xyz: Vec<[i32; 3]>,
    triangles: Vec<[u32; 3]>,
    parts: Vec<Part>,
}

impl Mesh {
    /// Welds `surfaces`, `points` holding each one's vertices on the grid, so that surfaces share
    /// the vertices they meet at. Vertices keep their order, and those no triangle names are dropped.
    fn weld(surfaces: &[Surface], points: &[Vec<[i32; 3]>]) -> Self {
        let mut ids: HashMap<[i32; 3], u32> = HashMap::new();
        let mut welded = Vec::new();
        let mut triangles = Vec::new();
        let mut parts = Vec::new();
        for (surface, points) in surfaces.iter().zip(points) {
            let local: Vec<u32> = points
                .iter()
                .map(|&p| {
                    *ids.entry(p).or_insert_with(|| {
                        welded.push(p);
                        u32::try_from(welded.len() - 1).unwrap_or(u32::MAX)
                    })
                })
                .collect();
            for t in &surface.triangles {
                let [a, b, c] = t.map(|i| local[i as usize]);
                if a != b && b != c && c != a {
                    triangles.push([a, b, c]);
                    parts.push(surface.part);
                }
            }
        }
        let mut used = vec![false; welded.len()];
        for &i in triangles.iter().flatten() {
            used[i as usize] = true;
        }
        let mut renumbered = vec![0; welded.len()];
        let mut xyz = Vec::new();
        for (i, p) in welded.into_iter().enumerate() {
            if used[i] {
                renumbered[i] = u32::try_from(xyz.len()).unwrap_or(u32::MAX);
                xyz.push(p);
            }
        }
        for t in &mut triangles {
            *t = t.map(|i| renumbered[i as usize]);
        }
        Self {
            xyz,
            triangles,
            parts,
        }
    }
}

/// `building` on the tile of each of `zooms` that holds the centre of its bounding box.
/// A zoom is left out where the building has no triangle or does not fit the tile's grid.
fn place(
    building: &Building,
    zooms: std::ops::RangeInclusive<u8>,
    z_step: ZStep,
) -> Vec<(Tile, Mesh)> {
    let positions = || building.surfaces.iter().flat_map(|s| &s.positions);
    let fold =
        |f: fn(f64, f64) -> f64, i: usize, init: f64| positions().map(|p| p[i]).fold(init, f);
    let [west, south, ground] = [0, 1, 2].map(|i| fold(f64::min, i, f64::INFINITY));
    let [east, north] = [0, 1].map(|i| fold(f64::max, i, f64::NEG_INFINITY));
    let centre = unit(f64::midpoint(west, east), f64::midpoint(south, north));
    let units: Vec<Vec<[f64; 2]>> = building
        .surfaces
        .iter()
        .map(|s| s.positions.iter().map(|p| unit(p[0], p[1])).collect())
        .collect();
    let mut placed = Vec::new();
    for zoom in zooms {
        let tile = Tile::of(centre, zoom);
        let extent = tile.extent();
        let points: Option<Vec<Vec<[i32; 3]>>> = building
            .surfaces
            .iter()
            .zip(&units)
            .map(|(surface, units)| {
                surface
                    .positions
                    .iter()
                    .zip(units)
                    .map(|(p, &u)| {
                        let [x, y] = tile.grid(u, extent)?;
                        Some([x, y, z_step.z(p[2] - ground)?])
                    })
                    .collect()
            })
            .collect();
        let Some(points) = points else {
            continue;
        };
        let mesh = Mesh::weld(&building.surfaces, &points);
        if !mesh.triangles.is_empty() {
            placed.push((tile, mesh));
        }
    }
    placed
}

/// The layer of a tile's meshes, each with its feature id and building.
fn layer(
    name: &str,
    tile: Tile,
    z_step: ZStep,
    meshes: Vec<(u64, &Building, Mesh)>,
) -> AnyResult<TileLayer> {
    let mut builder = TileLayer::builder(name, tile.extent())?;
    builder.set_z_step(z_step)?;
    let gml_id = builder.add_property("gml_id", PropKind::Str)?;
    let part = builder.add_scoped_m_value("part", PropKind::Str, MValueScope::Triangle)?;
    for (id, building, mesh) in meshes {
        let points = mesh.xyz.iter().map(|&[x, y, _]| Point::new(x, y)).collect();
        let mut feature = builder.feature(Geometry::MultiPoint(MultiPoint(points)));
        feature.id(Some(id));
        feature.z(mesh.xyz.iter().map(|v| v[2]).collect())?;
        feature.triangles(mesh.triangles)?;
        feature.property(
            gml_id,
            PropValue::Str(building.gml_id.as_deref().map(Into::into)),
        )?;
        let parts = mesh.parts.iter().map(|p| p.as_str().into()).collect();
        feature.m_value(part, MValue::Str(Some(parts)))?;
        feature.finish()?;
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPERA: [f64; 2] = [2.331_716_682_733_388, 48.872_017_348_646_814];

    fn opera() -> [f64; 2] {
        let [lon, lat] = OPERA.map(f64::to_radians);
        unit(lon, lat)
    }

    #[test]
    fn a_position_lies_in_the_tile_of_its_zoom_on_a_grid_of_about_a_centimetre() {
        let tile = Tile::of(opera(), 17);
        assert_eq!(
            tile,
            Tile {
                zoom: 17,
                x: 66384,
                y: 45083
            }
        );
        let extents = [15, 16, 17].map(|zoom| Tile::of(opera(), zoom).extent());
        assert_eq!(extents, [65536, 32768, 16384]);
        let [x, y] = tile.grid(opera(), tile.extent()).unwrap();
        assert!((0..16384).contains(&x) && (0..16384).contains(&y));
    }

    #[test]
    fn surfaces_share_their_welded_vertices_and_each_triangle_keeps_its_part() {
        let surface = |part, triangles| Surface {
            part,
            positions: Vec::new(),
            triangles,
        };
        let surfaces = [
            surface(Part::Wall, vec![[0, 1, 2]]),
            surface(Part::Roof, vec![[0, 2, 1], [0, 0, 1], [3, 3, 3]]),
        ];
        let points = [
            vec![[0, 0, 0], [5, 0, 0], [0, 5, 0]],
            vec![[5, 0, 0], [5, 5, 0], [0, 5, 0], [9, 9, 9]],
        ];
        let mesh = Mesh::weld(&surfaces, &points);
        assert_eq!(mesh.xyz, [[0, 0, 0], [5, 0, 0], [0, 5, 0], [5, 5, 0]]);
        assert_eq!(mesh.triangles, [[0, 1, 2], [1, 2, 3]]);
        assert_eq!(mesh.parts, [Part::Wall, Part::Roof]);
    }
}
