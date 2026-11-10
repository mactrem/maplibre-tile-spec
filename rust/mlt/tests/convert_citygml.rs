//! End-to-end tests of `mlt convert` tiling a `CityGML` file into a directory of v2 `Meshes` tiles.

// hotpath appends its profile to stderr, which the error checks read.
#![cfg(not(feature = "hotpath"))]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use mlt_core::geo_types::Geometry;
use mlt_core::{Decoder, MValue, Parser, PropValue, TileFeature, ZStep};

/// Two boxes in Hamburg: `tower`, 60 m high with a ground surface, and `kiosk`, 3 m high without.
const BUILDINGS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test/fixtures/meshes/buildings.gml"
);
const TILED: [&str; 6] = ["--mlt-version", "2", "--min-zoom", "15", "--max-zoom", "16"];

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// A temp directory deleted on drop.
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
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Run `mlt convert <input> <output> <args>`, returning its stderr when it fails.
fn convert(input: &Path, output: &Path, args: &[&str]) -> Result<(), String> {
    let out = Command::new(env!("CARGO_BIN_EXE_mlt"))
        .arg("convert")
        .arg(input)
        .arg(output)
        .args(args)
        .env_remove("RUST_BACKTRACE")
        .output()
        .unwrap();
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn each_building_becomes_one_mesh_per_zoom_with_parts_and_heights_above_ground() {
    let dir = TempDir::new();
    let output = dir.0.join("out");
    convert(Path::new(BUILDINGS), &output, &TILED).unwrap();
    assert_eq!(fs::read_dir(&output).unwrap().count(), 2);
    for tile in ["15/17294/10590", "16/34588/21180"] {
        let bytes = fs::read(output.join(format!("{tile}.mlt"))).unwrap();
        let mut layers = Parser::default().parse_layers(&bytes).unwrap();
        assert_eq!(layers.len(), 1, "{tile}");
        let layer = layers.pop().unwrap();
        let layer = layer.into_tile(&mut Decoder::default()).unwrap().unwrap();
        assert_eq!(layer.name(), "buildings");
        assert_eq!(layer.z_step(), Some(ZStep::above_ground(-2).unwrap()));
        assert_eq!(layer.property_names(), ["gml_id"]);
        assert_eq!(layer.m_value_names(), ["part"]);
        let mut features: Vec<&TileFeature> = layer.features().iter().collect();
        features.sort_by_key(|f| f.id());
        let [tower, kiosk] = features[..] else {
            panic!("{tile}: {} features", features.len());
        };
        assert_box(tower, "tower", 6000, [8, 2, 2]);
        assert_box(kiosk, "kiosk", 300, [8, 2, 0]);
    }
}

/// `feature` is the box `gml_id`: 8 welded corners at heights `0` and `top` in 1 cm steps above
/// its ground, and `[wall, roof, ground]` triangles, each counter-clockwise in (x, -y, z) seen
/// from outside and facing the way its part says.
fn assert_box(feature: &TileFeature, gml_id: &str, top: i32, parts: [usize; 3]) {
    assert_eq!(feature.properties(), [PropValue::Str(Some(gml_id.into()))]);
    let mut heights = feature.z().to_vec();
    heights.sort_unstable();
    heights.dedup();
    assert_eq!(heights, [0, top], "{gml_id}");
    let Geometry::MultiPoint(points) = feature.geometry() else {
        panic!("{gml_id}: {:?}", feature.geometry());
    };
    let xyz: Vec<[f64; 3]> = points
        .iter()
        .zip(feature.z())
        .map(|(p, &z)| [f64::from(p.x()), -f64::from(p.y()), f64::from(z)])
        .collect();
    assert_eq!(xyz.len(), 8, "{gml_id}");
    let MValue::Str(Some(names)) = &feature.m_values()[0] else {
        panic!("{gml_id}: {:?}", feature.m_values());
    };
    let count = |part: &str| names.iter().filter(|p| *p == part).count();
    let counts = [count("wall"), count("roof"), count("ground")];
    assert_eq!(counts, parts, "{gml_id}");

    let triangles = feature.triangles().unwrap();
    assert_eq!(triangles.len(), names.len(), "{gml_id}");
    let centre = [0, 1, 2].map(|i| xyz.iter().map(|p| p[i]).sum::<f64>() / 8.0);
    for (triangle, part) in triangles.iter().zip(names) {
        let [first, second, third] = triangle.map(|i| xyz[i as usize]);
        let [along, across] = [second, third].map(|c| [0, 1, 2].map(|i| c[i] - first[i]));
        let normal = [
            along[1] * across[2] - along[2] * across[1],
            along[2] * across[0] - along[0] * across[2],
            along[0] * across[1] - along[1] * across[0],
        ];
        let outwards: f64 = (0..3).map(|i| normal[i] * (first[i] - centre[i])).sum();
        assert!(outwards > 0.0, "{gml_id}: {triangle:?} faces inwards");
        let facing = match normal[2] {
            up if up > 0.0 => "roof",
            down if down < 0.0 => "ground",
            _ => "wall",
        };
        assert_eq!(part, facing, "{gml_id}: {triangle:?}");
    }
}

#[test]
fn citygml_input_rejects_flags_it_cannot_honour_and_srs_names_it_cannot_read() {
    let dir = TempDir::new();
    let input = Path::new(BUILDINGS);
    let output = dir.0.join("out");
    let archive = dir.0.join("out.pmtiles");
    let unknown_crs = dir.0.join("local.gml");
    let xml = fs::read_to_string(input).unwrap();
    fs::write(&unknown_crs, xml.replace("EPSG:25832", "urn:example:local")).unwrap();
    let verify = [&TILED[..], &["--verify"]].concat();
    for (input, output, args, expected) in [
        (input, &output, &TILED[2..], "needs --mlt-version 2"),
        (input, &output, &TILED[..4], "requires --max-zoom"),
        (
            input,
            &output,
            &verify[..],
            "--verify does not compare meshes",
        ),
        (
            input,
            &archive,
            &TILED[..],
            "archive output is not supported",
        ),
        (
            unknown_crs.as_path(),
            &output,
            &TILED[..],
            "names no EPSG code",
        ),
    ] {
        let err = convert(input, output, args).unwrap_err();
        assert!(err.contains(expected), "{expected}: {err}");
    }
    assert!(!output.exists() && !archive.exists());
}
