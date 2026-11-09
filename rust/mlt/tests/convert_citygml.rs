//! End-to-end tests of `mlt convert` tiling a `CityGML` file into a directory of v2 `Meshes` tiles.

// hotpath appends its profile to stderr, which the error checks read.
#![cfg(not(feature = "hotpath"))]

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use mlt_core::geo_types::Geometry;
use mlt_core::{Decoder, MValue, Parser, PropValue, TileLayer, ZStep};
use walkdir::WalkDir;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mlt-convert-citygml-test-{}-{id}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn mlt(args: &[&dyn AsRef<std::ffi::OsStr>]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mlt"))
        .arg("convert")
        .args(args.iter().map(AsRef::as_ref))
        .env_remove("RUST_BACKTRACE")
        .output()
        .unwrap()
}

fn succeeds(out: &Output) {
    assert!(
        out.status.success(),
        "mlt convert failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn fails(out: &Output) -> String {
    assert!(!out.status.success(), "mlt convert unexpectedly succeeded");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// A LOD2 box of walls, a roof and a ground surface, in UTM zone 32 metres, rings wound
/// counter-clockwise seen from outside.
fn building(id: &str, [x0, y0]: [f64; 2], size: f64, height: f64, ground: f64) -> String {
    let (x1, y1, g, top) = (x0 + size, y0 + size, ground, ground + height);
    let ring = |points: &[[f64; 3]]| {
        let mut list = String::new();
        for p in points.iter().chain(&points[..1]) {
            write!(list, "{} {} {} ", p[0], p[1], p[2]).unwrap();
        }
        format!(
            "<gml:surfaceMember><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>{}</gml:posList>\
             </gml:LinearRing></gml:exterior></gml:Polygon></gml:surfaceMember>",
            list.trim_end()
        )
    };
    let surface = |semantic: &str, rings: &[String]| {
        format!(
            "<bldg:boundedBy><bldg:{semantic}><bldg:lod2MultiSurface><gml:MultiSurface>{}\
             </gml:MultiSurface></bldg:lod2MultiSurface></bldg:{semantic}></bldg:boundedBy>",
            rings.concat()
        )
    };
    let walls = [
        ring(&[[x0, y0, g], [x1, y0, g], [x1, y0, top], [x0, y0, top]]),
        ring(&[[x1, y0, g], [x1, y1, g], [x1, y1, top], [x1, y0, top]]),
        ring(&[[x1, y1, g], [x0, y1, g], [x0, y1, top], [x1, y1, top]]),
        ring(&[[x0, y1, g], [x0, y0, g], [x0, y0, top], [x0, y1, top]]),
    ];
    let roof = [ring(&[
        [x0, y0, top],
        [x1, y0, top],
        [x1, y1, top],
        [x0, y1, top],
    ])];
    let floor = [ring(&[[x0, y0, g], [x0, y1, g], [x1, y1, g], [x1, y0, g]])];
    format!(
        r#"<core:cityObjectMember><bldg:Building gml:id="{id}">{}{}{}</bldg:Building></core:cityObjectMember>"#,
        surface("WallSurface", &walls),
        surface("RoofSurface", &roof),
        surface("GroundSurface", &floor),
    )
}

/// A 20 m box, 12 m high on ground 6.5 m above sea level, in Hamburg.
fn write_citygml(dir: &TempDir, srs_name: &str) -> PathBuf {
    let path = dir.join("buildings.gml");
    let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
        <core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
         xmlns:gml="http://www.opengis.net/gml">
        <gml:boundedBy><gml:Envelope srsName="{srs_name}" srsDimension="3"><gml:lowerCorner>566000 5934000 6.5</gml:lowerCorner>
        <gml:upperCorner>566020 5934020 18.5</gml:upperCorner></gml:Envelope></gml:boundedBy>
        {}</core:CityModel>"#,
        building("DEHH_B", [566_000.0, 5_934_000.0], 20.0, 12.0, 6.5),
    );
    fs::write(&path, xml).unwrap();
    path
}

fn decode(bytes: &[u8]) -> Vec<TileLayer> {
    let mut decoder = Decoder::default();
    Parser::default()
        .parse_layers(bytes)
        .unwrap()
        .into_iter()
        .map(|layer| layer.into_tile(&mut decoder).unwrap().unwrap())
        .collect()
}

/// Every `.mlt` under `dir` as `z/x/y`, sorted.
fn tile_paths(dir: &Path) -> Vec<String> {
    let mut paths: Vec<String> = WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "mlt"))
        .map(|e| {
            let rel = e.path().strip_prefix(dir).unwrap().with_extension("");
            rel.to_string_lossy().replace('\\', "/")
        })
        .collect();
    paths.sort();
    paths
}

#[test]
fn a_citygml_file_tiles_into_one_mesh_per_zoom_with_parts_and_heights_above_ground() {
    let dir = TempDir::new();
    let input = write_citygml(&dir, "EPSG:25832");
    let output = dir.join("out");
    succeeds(&mlt(&[
        &input,
        &output,
        &"--mlt-version",
        &"2",
        &"--min-zoom",
        &"15",
        &"--max-zoom",
        &"16",
    ]));

    let paths = tile_paths(&output);
    assert_eq!(paths.len(), 2, "{paths:?}");
    assert!(paths[0].starts_with("15/") && paths[1].starts_with("16/"));
    for path in &paths {
        let layers = decode(&fs::read(output.join(format!("{path}.mlt"))).unwrap());
        assert_eq!(layers.len(), 1);
        let layer = &layers[0];
        assert_eq!(layer.name(), "buildings");
        assert_eq!(layer.z_step(), Some(ZStep::above_ground(-2).unwrap()));
        assert_eq!(layer.property_names(), ["gml_id"]);
        assert_eq!(layer.m_value_names(), ["part"]);
        let [feature] = layer.features() else {
            panic!("{path}: {} features", layer.features().len());
        };
        assert_eq!(
            feature.properties(),
            [PropValue::Str(Some("DEHH_B".into()))]
        );

        // The 8 corners of the box, welded, 1 cm steps above its ground.
        let mut heights = feature.z().to_vec();
        heights.sort_unstable();
        heights.dedup();
        assert_eq!(heights, [0, 1200]);
        assert_eq!(feature.z().len(), 8);

        let triangles = feature.triangles().unwrap();
        assert_eq!(triangles.len(), 12);
        let MValue::Str(Some(parts)) = &feature.m_values()[0] else {
            panic!("{:?}", feature.m_values());
        };
        let count = |part: &str| parts.iter().filter(|p| *p == part).count();
        assert_eq!([count("wall"), count("roof"), count("ground")], [8, 2, 2]);

        // Each triangle winds counter-clockwise in (x, -y, z) seen from outside the box, and
        // its part is the side of the box it faces.
        let Geometry::MultiPoint(points) = feature.geometry() else {
            panic!("{:?}", feature.geometry());
        };
        let xyz: Vec<[f64; 3]> = points
            .iter()
            .zip(feature.z())
            .map(|(p, &z)| [f64::from(p.x()), -f64::from(p.y()), f64::from(z)])
            .collect();
        let centre = [0, 1, 2].map(|i| xyz.iter().map(|p| p[i]).sum::<f64>() / 8.0);
        for (triangle, part) in triangles.iter().zip(parts) {
            let corners = triangle.map(|i| xyz[i as usize]);
            let normal = normal(corners);
            let outwards: f64 = (0..3)
                .map(|i| normal[i] * (corners[0][i] - centre[i]))
                .sum();
            assert!(
                outwards > 0.0,
                "{path}: triangle {triangle:?} faces inwards"
            );
            let facing = match normal[2] {
                up if up > 0.0 => "roof",
                down if down < 0.0 => "ground",
                _ => "wall",
            };
            assert_eq!(part, facing, "{path}: triangle {triangle:?}");
        }
    }
}

/// The normal of the counter-clockwise triangle `corners`, as long as twice its area.
fn normal([first, second, third]: [[f64; 3]; 3]) -> [f64; 3] {
    let along = [0, 1, 2].map(|i| second[i] - first[i]);
    let across = [0, 1, 2].map(|i| third[i] - first[i]);
    [
        along[1] * across[2] - along[2] * across[1],
        along[2] * across[0] - along[0] * across[2],
        along[0] * across[1] - along[1] * across[0],
    ]
}

#[test]
fn citygml_input_rejects_flags_it_cannot_honour() {
    let dir = TempDir::new();
    let input = write_citygml(&dir, "EPSG:25832");
    let output = dir.join("out");
    let zooms: [&dyn AsRef<std::ffi::OsStr>; 2] = [&"--max-zoom", &"16"];
    let v2: [&dyn AsRef<std::ffi::OsStr>; 2] = [&"--mlt-version", &"2"];

    let err = fails(&mlt(&[&input, &output, &zooms[0], &zooms[1]]));
    assert!(err.contains("needs --mlt-version 2"), "{err}");
    let err = fails(&mlt(&[&input, &output, &v2[0], &v2[1]]));
    assert!(err.contains("requires --max-zoom"), "{err}");
    let err = fails(&mlt(&[
        &input,
        &output,
        &v2[0],
        &v2[1],
        &zooms[0],
        &zooms[1],
        &"--verify",
    ]));
    assert!(err.contains("--verify does not compare meshes"), "{err}");
    let archive = dir.join("out.pmtiles");
    let err = fails(&mlt(&[
        &input, &archive, &v2[0], &v2[1], &zooms[0], &zooms[1],
    ]));
    assert!(err.contains("archive output is not supported"), "{err}");
    assert!(!output.exists() && !archive.exists());
}

#[test]
fn citygml_input_without_a_known_crs_fails() {
    let dir = TempDir::new();
    let input = write_citygml(&dir, "urn:example:local");
    let output = dir.join("out");
    let err = fails(&mlt(&[
        &input,
        &output,
        &"--mlt-version",
        &"2",
        &"--max-zoom",
        &"15",
    ]));
    assert!(err.contains("names no EPSG code"), "{err}");
}
