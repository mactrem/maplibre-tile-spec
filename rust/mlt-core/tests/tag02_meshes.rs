//! Round-trip and wire-level tests for v2 meshes: the `Meshes` layout (`0xE`), the
//! `Mesh` geometry type and z as a height above the ground.

use insta::assert_snapshot;
use mlt_core::dump::{DumpTree, RenderOpts, annotate_tile, render};
use mlt_core::encoder::{EncoderConfig, Network, Profile, Tessellation, WireVersion};
use mlt_core::geo_types::{Geometry, MultiPoint, Point};
use mlt_core::geojson::FeatureCollection;
use mlt_core::{
    Decoder, GeometryType, Layer, Layer02, MValue, MValueScope, Parser, PropKind, PropValue,
    TileFeature, TileLayer, ZDatum, ZStep,
};
use proptest::prelude::*;
use rstest::rstest;

fn annotate(bytes: &[u8]) -> DumpTree {
    let (tree, err) = annotate_tile(bytes);
    assert!(err.is_none(), "annotate_tile: {err:?}");
    tree
}

fn cfg_v2() -> EncoderConfig {
    EncoderConfig::from_profile(Profile::Native(Network::Cellular4G))
        .with_wire_version(WireVersion::V02)
}

fn try_decode(bytes: &[u8]) -> mlt_core::MltResult<TileLayer> {
    only_v2_layer(Parser::default().parse_layers(bytes)?).into_tile(&mut Decoder::default())
}

fn only_v2_layer(layers: Vec<Layer<'_>>) -> Layer02<'_> {
    assert_eq!(layers.len(), 1);
    let Layer::Tag02(layer) = layers.into_iter().next().expect("a layer") else {
        panic!("expected a v2 layer")
    };
    layer
}

fn decode(bytes: &[u8]) -> TileLayer {
    try_decode(bytes).expect("decode")
}

/// The leaf regions of the dump tile `bytes` without gaps, and render.
fn assert_dump_covers(bytes: &[u8]) {
    let tree = annotate(bytes);
    let mut leaves: Vec<(usize, usize)> = tree
        .regions
        .iter()
        .filter(|r| !r.container)
        .map(|r| (r.offset, r.len))
        .collect();
    leaves.sort_unstable();
    let mut cursor = 0;
    for (offset, len) in &leaves {
        assert_eq!(*offset, cursor, "gap/overlap at offset {offset}");
        cursor += len;
    }
    assert_eq!(cursor, bytes.len());
    render(&tree, bytes, &RenderOpts::default(), &mut Vec::new()).expect("render");
}

/// The meaning of each bit field of the byte the dump labels `label`.
fn byte_meanings(bytes: &[u8], label: &str) -> Vec<String> {
    annotate(bytes)
        .regions
        .iter()
        .filter(|r| r.label == label)
        .flat_map(|r| r.bits.iter().map(|b| b.meaning().to_string()))
        .collect()
}

/// Where the dump puts the byte it labels `label`.
fn byte_offset(bytes: &[u8], label: &str) -> usize {
    annotate(bytes)
        .regions
        .iter()
        .find(|r| r.label == label)
        .map(|r| r.offset)
        .expect("a labelled byte")
}

fn rotations([a, b, c]: [u32; 3]) -> [[u32; 3]; 3] {
    [[a, b, c], [b, c, a], [c, a, b]]
}

/// Triangles each rotated to start at its smallest corner, then sorted,
/// which is all of them the triangle trees promise to keep.
fn canonical(triangles: &[[u32; 3]]) -> Vec<[u32; 3]> {
    let mut out: Vec<[u32; 3]> = triangles
        .iter()
        .map(|&t| rotations(t).into_iter().min().unwrap())
        .collect();
    out.sort_unstable();
    out
}

/// Everything a feature holds, with its triangles in canonical form.
type FeatureRow = (
    Option<u64>,
    Geometry<i32>,
    Vec<PropValue>,
    Vec<MValue>,
    Vec<i32>,
    Option<Vec<[u32; 3]>>,
);

/// The layer's features by id, each as a [`FeatureRow`], so the encoder may reorder them.
fn rows(layer: &TileLayer) -> Vec<FeatureRow> {
    let mut rows: Vec<FeatureRow> = layer
        .features()
        .iter()
        .map(|f| {
            (
                f.id(),
                f.geometry().clone(),
                f.properties().to_vec(),
                f.m_values().to_vec(),
                f.z().to_vec(),
                f.triangles().map(canonical),
            )
        })
        .collect();
    rows.sort_by_key(|row| row.0);
    rows
}

/// Encode with `cfg`, decode, and check every feature came back whole.
fn assert_round_trips(layer: &TileLayer, cfg: EncoderConfig) -> Vec<u8> {
    let bytes = layer.encode(cfg).expect("v2 encode");
    let back = decode(&bytes);
    assert_eq!(back.name(), layer.name());
    assert_eq!(back.extent(), layer.extent());
    assert_eq!(back.z_step(), layer.z_step());
    assert_eq!(back.property_names(), layer.property_names());
    assert_eq!(back.m_value_names(), layer.m_value_names());
    assert_eq!(rows(&back), rows(layer));
    assert_dump_covers(&bytes);
    bytes
}

fn height() -> ZStep {
    ZStep::above_ground(-2).unwrap()
}

/// One mesh: `(x, y, z)` vertices, triangles over them, and a part per vertex.
struct Mesh {
    id: u64,
    vertices: Vec<[i32; 3]>,
    triangles: Vec<[u32; 3]>,
    parts: Option<Vec<&'static str>>,
}

fn landmarks(meshes: &[Mesh]) -> TileLayer {
    let mut builder = TileLayer::builder("landmarks", 16384).unwrap();
    builder.set_z_step(height()).unwrap();
    let name = builder.add_property("name", PropKind::Str).unwrap();
    let part = builder.add_m_value("part", PropKind::Str).unwrap();
    let color = builder.add_m_value("color", PropKind::U32).unwrap();
    for mesh in meshes {
        let points = mesh.vertices.iter().map(|&[x, y, _]| Point::new(x, y));
        let mut feature = builder.feature(Geometry::MultiPoint(points.collect()));
        feature.id(Some(mesh.id));
        feature
            .z(mesh.vertices.iter().map(|v| v[2]).collect())
            .unwrap();
        feature.triangles(mesh.triangles.clone()).unwrap();
        feature
            .property(
                name,
                PropValue::Str(Some(format!("mesh {}", mesh.id).into())),
            )
            .unwrap();
        if let Some(parts) = &mesh.parts {
            let values = parts.iter().map(|&p| p.into()).collect();
            feature.m_value(part, MValue::Str(Some(values))).unwrap();
            let colors = parts
                .iter()
                .map(|p| if *p == "roof" { 0x8e_9b_a6 } else { 0xe6_dc_cb });
            feature
                .m_value(color, MValue::U32(Some(colors.collect())))
                .unwrap();
        }
        feature.finish().unwrap();
    }
    builder.finish()
}

/// A box of 8 vertices and 12 triangles, each facing out in (east, north, up).
fn cube(id: u64, x: i32, y: i32) -> Mesh {
    let s = 100;
    // Tile y runs south, so north is -y.
    let vertices = vec![
        [x, y, 0],
        [x + s, y, 0],
        [x + s, y - s, 0],
        [x, y - s, 0],
        [x, y, 1500],
        [x + s, y, 1500],
        [x + s, y - s, 1500],
        [x, y - s, 1500],
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    let parts = vec![
        "ground", "ground", "ground", "ground", "roof", "roof", "roof", "roof",
    ];
    Mesh {
        id,
        vertices,
        triangles,
        parts: Some(parts),
    }
}

#[test]
fn a_mesh_layer_round_trips() {
    let layer = landmarks(&[cube(10, 0, 0), cube(20, 5000, -300)]);
    let bytes = assert_round_trips(&layer, cfg_v2());
    assert_eq!(
        byte_meanings(&bytes, "layout").last().unwrap(),
        "geometry layout = Meshes"
    );
    assert!(
        byte_meanings(&bytes, "header")
            .contains(&"every feature is a Mesh, no types stream".to_string())
    );
}

#[test]
fn a_cube_keeps_its_triangles_up_to_order_and_rotation() {
    let mesh = cube(1, 10, 10);
    let expected = mesh.triangles.clone();
    let back = decode(&landmarks(&[mesh]).encode(cfg_v2()).unwrap());
    let feature = &back.features()[0];
    assert_eq!(
        canonical(feature.triangles().unwrap()),
        canonical(&expected)
    );
    // The trees keep each triangle's winding: no triangle comes back reversed.
    for &t in feature.triangles().unwrap() {
        assert!(rotations(t).iter().any(|r| expected.contains(r)), "{t:?}");
    }
}

#[test]
fn meshes_survive_every_sort_the_encoder_tries() {
    let meshes: Vec<Mesh> = (0..40_i32)
        .map(|i| {
            cube(
                1000 - u64::from(i.unsigned_abs()),
                (i % 7) * 900,
                (i / 7) * -900,
            )
        })
        .collect();
    let layer = landmarks(&meshes);
    assert_round_trips(&layer, cfg_v2());
    assert_round_trips(
        &layer,
        cfg_v2()
            .with_spatial_morton_sort(false)
            .with_spatial_hilbert_sort(false)
            .with_id_sort(false),
    );
}

#[test]
fn tessellation_settings_leave_a_mesh_layer_alone() {
    let layer = landmarks(&[cube(1, 0, 0), cube(2, 400, 0)]);
    for tessellation in [Tessellation::Always, Tessellation::Off] {
        let bytes = assert_round_trips(&layer, cfg_v2().with_tessellation(tessellation));
        assert_eq!(
            byte_meanings(&bytes, "layout").last().unwrap(),
            "geometry layout = Meshes"
        );
    }
    assert_round_trips(&layer, cfg_v2().with_triangles_only(true));
}

#[test]
fn an_empty_mesh_and_an_unnamed_vertex_round_trip() {
    let empty = Mesh {
        id: 1,
        vertices: vec![],
        triangles: vec![],
        parts: None,
    };
    let no_triangles = Mesh {
        id: 2,
        vertices: vec![[1, 2, 3], [4, 5, 6]],
        triangles: vec![],
        parts: Some(vec!["wall", "wall"]),
    };
    let spare_vertex = Mesh {
        id: 3,
        vertices: vec![[0, 0, 0], [10, 0, 0], [0, -10, 0], [99, 99, 99]],
        triangles: vec![[0, 1, 2]],
        parts: Some(vec!["roof", "roof", "roof", "lamp"]),
    };
    assert_round_trips(&landmarks(&[empty, no_triangles, spare_vertex]), cfg_v2());
}

#[test]
fn a_layer_of_meshes_without_triangles_still_writes_the_meshes_layout() {
    let layer = landmarks(&[Mesh {
        id: 1,
        vertices: vec![[1, 2, 3]],
        triangles: vec![],
        parts: None,
    }]);
    let bytes = assert_round_trips(&layer, cfg_v2());
    assert_eq!(
        byte_meanings(&bytes, "layout").last().unwrap(),
        "geometry layout = Meshes"
    );
}

#[test]
fn degenerate_and_repeated_triangles_round_trip() {
    let mesh = Mesh {
        id: 7,
        vertices: vec![[0, 0, 0], [10, 0, 0], [0, -10, 0], [10, -10, 5]],
        triangles: vec![
            [0, 1, 2],
            [0, 1, 2],
            [2, 1, 0],
            [0, 0, 1],
            [3, 3, 3],
            [1, 3, 2],
        ],
        parts: None,
    };
    assert_round_trips(&landmarks(&[mesh]), cfg_v2());
}

#[test]
fn heights_decode_without_the_terrain_rgb_offset() {
    let back = decode(&landmarks(&[cube(1, 0, 0)]).encode(cfg_v2()).unwrap());
    let step = back.z_step().unwrap();
    assert_eq!(step.datum(), ZDatum::Ground);
    let roof = back.features()[0].z().iter().copied().max().unwrap();
    assert_eq!(step.elevation(roof).to_bits(), 15.0_f64.to_bits());
}

#[test]
fn geojson_writes_a_mesh_as_a_multi_point_with_triangles() {
    let mesh = Mesh {
        id: 5,
        vertices: vec![[0, 0, 0], [10, 0, 0], [0, -10, 250]],
        triangles: vec![[0, 1, 2]],
        parts: Some(vec!["wall", "wall", "roof"]),
    };
    let bytes = landmarks(&[mesh]).encode(cfg_v2()).unwrap();
    let layers = Decoder::default()
        .decode_all(Parser::default().parse_layers(&bytes).unwrap())
        .unwrap();
    let json = serde_json::to_string(&FeatureCollection::from_layers(layers).unwrap()).unwrap();
    assert_snapshot!(json, @r#"{"type":"FeatureCollection","features":[{"type":"Feature","id":5,"properties":{"_extent":16384,"_layer":"landmarks","_z_datum":"ground","_z_step":-2,"m:color":[15129803,15129803,9345958],"m:part":["wall","wall","roof"],"name":"mesh 5"},"geometry":{"type":"MultiPoint","coordinates":[[0,0,0],[10,0,0],[0,-10,250]]},"triangles":[[0,1,2]]}]}"#);
}

fn arb_mesh() -> impl Strategy<Value = (Vec<[i32; 3]>, Vec<[u32; 3]>)> {
    prop::collection::vec(any::<[i32; 3]>(), 0..24).prop_flat_map(|vertices| {
        let n = u32::try_from(vertices.len()).unwrap();
        let triangles = if n == 0 {
            Just(Vec::new()).boxed()
        } else {
            prop::collection::vec(prop::array::uniform3(0..n), 0..40).boxed()
        };
        (Just(vertices), triangles)
    })
}

proptest! {
    #[test]
    fn random_meshes_round_trip(meshes in prop::collection::vec(arb_mesh(), 1..6)) {
        let mut builder = TileLayer::builder("random", 4096).unwrap();
        builder.set_z_step(height()).unwrap();
        let key = builder.add_m_value("m", PropKind::I32).unwrap();
        for (id, (vertices, triangles)) in meshes.iter().enumerate() {
            let points: MultiPoint<i32> = vertices.iter().map(|&[x, y, _]| Point::new(x, y)).collect();
            let mut feature = builder.feature(Geometry::MultiPoint(points));
            feature.id(Some(id as u64));
            feature.z(vertices.iter().map(|v| v[2]).collect()).unwrap();
            feature.triangles(triangles.clone()).unwrap();
            let m = vertices.iter().map(|v| v[0] ^ v[2]).collect();
            feature.m_value(key, MValue::I32(Some(m))).unwrap();
            feature.finish().unwrap();
        }
        let layer = builder.finish();
        let bytes = layer.encode(cfg_v2()).expect("v2 encode");
        let back = decode(&bytes);
        prop_assert_eq!(back.z_step(), layer.z_step());
        prop_assert_eq!(rows(&back), rows(&layer));
    }
}

#[test]
fn v1_rejects_a_mesh_layer() {
    let mut builder = TileLayer::builder("landmarks", 4096).unwrap();
    builder.set_z_step(height()).unwrap();
    let mut mesh = builder.feature(Geometry::MultiPoint(MultiPoint(vec![Point::new(0, 0)])));
    mesh.z(vec![0]).unwrap().triangles(vec![[0, 0, 0]]).unwrap();
    mesh.finish().unwrap();
    let err = builder
        .finish()
        .encode(cfg_v2().with_wire_version(WireVersion::V01))
        .unwrap_err();
    assert_snapshot!(err, @"meshes are a v2 feature, so layer landmarks cannot be written as v1");
}

#[test]
fn a_mesh_needs_a_z_step() {
    let mut layer = TileLayer::new("flat", 4096).unwrap();
    let mut mesh = TileFeature::new(Geometry::MultiPoint(MultiPoint(vec![Point::new(0, 0)])));
    mesh.set_triangles(vec![[0, 0, 0]]).unwrap();
    layer.push_feature(mesh).unwrap();
    let err = layer.encode(cfg_v2()).unwrap_err();
    assert_snapshot!(err, @"a mesh needs (x, y, z) vertices, so its layer needs a z step");
}

#[test]
fn a_layer_cannot_mix_meshes_with_other_geometries() {
    let mut builder = TileLayer::builder("mixed", 4096).unwrap();
    builder.set_z_step(height()).unwrap();
    let mut mesh = builder.feature(Geometry::MultiPoint(MultiPoint(vec![Point::new(0, 0)])));
    mesh.z(vec![0]).unwrap().triangles(vec![[0, 0, 0]]).unwrap();
    mesh.finish().unwrap();
    let mut point = builder.feature(Geometry::Point(Point::new(1, 1)));
    point.z(vec![0]).unwrap();
    point.finish().unwrap();
    let err = builder.finish().encode(cfg_v2()).unwrap_err();
    assert_snapshot!(err, @"a layer cannot mix meshes with other geometries");
}

#[rstest]
#[case::multi_polygon_over_meshes(
    0x6,
    "uniform geometry type Some(MultiPolygon) cannot go with geometry layout Meshes: a Mesh needs the Meshes layout, which holds nothing else"
)]
#[case::types_stream_over_meshes(
    0x0,
    "uniform geometry type None cannot go with geometry layout Meshes: a Mesh needs the Meshes layout, which holds nothing else"
)]
fn a_meshes_layout_needs_the_uniform_type_mesh(#[case] nibble: u8, #[case] expected: &str) {
    let mut bytes = landmarks(&[cube(1, 0, 0)]).encode(cfg_v2()).unwrap();
    let header = byte_offset(&bytes, "header");
    bytes[header] = (bytes[header] & 0b1000_1111) | (nibble << 4);
    assert_eq!(try_decode(&bytes).unwrap_err().to_string(), expected);
}

#[test]
fn the_uniform_type_mesh_needs_the_meshes_layout() {
    let mut builder = TileLayer::builder("points", 4096).unwrap();
    builder.set_z_step(height()).unwrap();
    for i in 0..3 {
        let mut feature = builder.feature(Geometry::MultiPoint(MultiPoint(vec![Point::new(i, i)])));
        feature.z(vec![i]).unwrap();
        feature.finish().unwrap();
    }
    let mut bytes = builder.finish().encode(cfg_v2()).unwrap();
    assert_eq!(
        byte_meanings(&bytes, "layout").last().unwrap(),
        "geometry layout = MultiPoints"
    );
    let header = byte_offset(&bytes, "header");
    bytes[header] |= 0b0111_0000;
    assert_snapshot!(
        try_decode(&bytes).unwrap_err(),
        @"uniform geometry type Some(Mesh) cannot go with geometry layout MultiPoints: a Mesh needs the Meshes layout, which holds nothing else"
    );
}

/// Each triangle in canonical rotation next to its value, sorted, which is what an encoder
/// must keep of a triangle-scoped column.
fn paired<T: Ord + Clone>(triangles: &[[u32; 3]], values: &[T]) -> Vec<([u32; 3], T)> {
    assert_eq!(triangles.len(), values.len(), "a value per triangle");
    let mut out: Vec<([u32; 3], T)> = triangles
        .iter()
        .map(|&t| canonical(&[t])[0])
        .zip(values.iter().cloned())
        .collect();
    out.sort();
    out
}

fn strs(values: &MValue) -> Vec<String> {
    let MValue::Str(Some(values)) = values else {
        panic!("{values:?}")
    };
    values.iter().map(ToString::to_string).collect()
}

/// Vertices, triangles, a part per triangle and an occlusion per vertex.
type Example = (
    &'static [[i32; 3]],
    &'static [[u32; 3]],
    &'static [&'static str],
    &'static [u8],
);

/// A tetrahedron and a quad with a triangle-scoped `part` and a vertex-scoped `ao`. The
/// tetrahedron's triangles are given roof first, not in the order its trees decode them.
fn tetrahedron_and_quad() -> TileLayer {
    let mut builder = TileLayer::builder("landmarks", 4096).unwrap();
    builder.set_z_step(height()).unwrap();
    let part = builder
        .add_scoped_m_value("part", PropKind::Str, MValueScope::Triangle)
        .unwrap();
    let ao = builder.add_m_value("ao", PropKind::U8).unwrap();
    let meshes: [Example; 2] = [
        (
            &[[0, 0, 0], [100, 0, 0], [0, -100, 0], [30, -30, 100]],
            &[[3, 2, 0], [0, 1, 3], [0, 2, 1], [1, 2, 3]],
            &["roof", "roof", "wall", "wall"],
            &[0, 0, 9, 9],
        ),
        (
            &[[500, 0, 0], [600, 0, 0], [600, -100, 0], [500, -100, 0]],
            &[[0, 1, 2], [0, 2, 3]],
            &["window", "window"],
            &[4, 4, 8, 8],
        ),
    ];
    for (id, (vertices, triangles, parts, occlusion)) in meshes.into_iter().enumerate() {
        let points = vertices.iter().map(|&[x, y, _]| Point::new(x, y));
        let mut feature = builder.feature(Geometry::MultiPoint(points.collect()));
        feature.id(Some(id as u64));
        feature.z(vertices.iter().map(|v| v[2]).collect()).unwrap();
        feature.triangles(triangles.to_vec()).unwrap();
        let parts = parts.iter().map(|&p| p.into()).collect();
        feature.m_value(part, MValue::Str(Some(parts))).unwrap();
        feature
            .m_value(ao, MValue::U8(Some(occlusion.to_vec())))
            .unwrap();
        feature.finish().unwrap();
    }
    builder.finish()
}

#[test]
fn triangle_scoped_values_stay_with_their_triangles() {
    let layer = tetrahedron_and_quad();
    let bytes = layer.encode(cfg_v2()).unwrap();
    assert_dump_covers(&bytes);
    let columns: Vec<String> = annotate(&bytes)
        .regions
        .iter()
        .filter(|r| r.container && r.label.starts_with("m_value"))
        .map(|r| r.label.clone())
        .collect();
    assert_eq!(
        columns,
        [
            r#"m_value[0] Str "part" per triangle"#,
            r#"m_value[1] U8 "ao" per vertex"#
        ]
    );

    let back = decode(&bytes);
    assert_eq!(
        back.m_value_scopes(),
        [MValueScope::Triangle, MValueScope::Vertex]
    );
    for (sent, got) in layer.features().iter().zip(back.features()) {
        assert_eq!(got.id(), sent.id());
        assert_eq!(got.z(), sent.z());
        assert_eq!(got.m_values()[1], sent.m_values()[1], "ao stays per vertex");
        assert_eq!(
            paired(got.triangles().unwrap(), &strs(&got.m_values()[0])),
            paired(sent.triangles().unwrap(), &strs(&sent.m_values()[0])),
        );
    }
}

#[test]
fn triangle_scoped_values_follow_the_triangles_through_every_sort() {
    let meshes: Vec<Mesh> = (0..30_i32)
        .map(|i| {
            cube(
                100 - u64::from(i.unsigned_abs()),
                (i % 6) * 700,
                (i / 6) * -700,
            )
        })
        .collect();
    let mut builder = TileLayer::builder("landmarks", 16384).unwrap();
    builder.set_z_step(height()).unwrap();
    let key = builder
        .add_scoped_m_value("face", PropKind::I32, MValueScope::Triangle)
        .unwrap();
    for mesh in &meshes {
        let points = mesh.vertices.iter().map(|&[x, y, _]| Point::new(x, y));
        let mut feature = builder.feature(Geometry::MultiPoint(points.collect()));
        feature.id(Some(mesh.id));
        feature
            .z(mesh.vertices.iter().map(|v| v[2]).collect())
            .unwrap();
        feature.triangles(mesh.triangles.clone()).unwrap();
        let faces = (0..12).map(|t| i32::try_from(mesh.id).unwrap() * 100 + t);
        feature
            .m_value(key, MValue::I32(Some(faces.collect())))
            .unwrap();
        feature.finish().unwrap();
    }
    let layer = builder.finish();
    let faces = |feature: &TileFeature| {
        let MValue::I32(Some(faces)) = &feature.m_values()[0] else {
            panic!("{:?}", feature.m_values())
        };
        (feature.id(), paired(feature.triangles().unwrap(), faces))
    };
    let mut sent: Vec<_> = layer.features().iter().map(faces).collect();
    sent.sort();
    for cfg in [
        cfg_v2(),
        cfg_v2()
            .with_spatial_morton_sort(false)
            .with_spatial_hilbert_sort(false)
            .with_id_sort(false),
    ] {
        let bytes = layer.encode(cfg).unwrap();
        assert_dump_covers(&bytes);
        let mut got: Vec<_> = decode(&bytes).features().iter().map(faces).collect();
        got.sort();
        assert_eq!(got, sent);
    }
}

#[test]
fn a_reserved_scope_byte_is_rejected() {
    let mut bytes = tetrahedron_and_quad().encode(cfg_v2()).unwrap();
    let scope = byte_offset(&bytes, "scope");
    assert_eq!(bytes[scope], 1);
    bytes[scope] = 2;
    assert_snapshot!(
        try_decode(&bytes).unwrap_err(),
        @"m-value scope byte 2 is reserved: only 0 (vertex) and 1 (triangle) are defined"
    );
}

#[test]
fn a_triangle_scoped_column_needs_a_value_per_triangle() {
    let mut builder = TileLayer::builder("landmarks", 4096).unwrap();
    builder.set_z_step(height()).unwrap();
    let key = builder
        .add_scoped_m_value("part", PropKind::U8, MValueScope::Triangle)
        .unwrap();
    let mut mesh = builder.feature(Geometry::MultiPoint(MultiPoint(vec![
        Point::new(0, 0),
        Point::new(1, 0),
        Point::new(0, 1),
    ])));
    mesh.z(vec![0, 0, 0]).unwrap();
    mesh.triangles(vec![[0, 1, 2]]).unwrap();
    mesh.m_value(key, MValue::U8(Some(vec![1, 2, 3]))).unwrap();
    assert_snapshot!(
        mesh.finish().unwrap_err(),
        @"m-value 0 holds 3 values, but the mesh has 1 triangles"
    );
}

#[test]
fn only_a_mesh_has_triangle_scoped_values() {
    let mut builder = TileLayer::builder("lines", 4096).unwrap();
    let key = builder
        .add_scoped_m_value("part", PropKind::U8, MValueScope::Triangle)
        .unwrap();
    let mut line = builder.feature(Geometry::MultiPoint(MultiPoint(vec![
        Point::new(0, 0),
        Point::new(1, 0),
    ])));
    line.m_value(key, MValue::U8(Some(vec![1]))).unwrap();
    assert_snapshot!(
        line.finish().unwrap_err(),
        @"m-value column part holds a value per triangle, which only a mesh has"
    );
}

#[test]
fn decoded_meshes_report_their_type() {
    let bytes = landmarks(&[cube(1, 0, 0)]).encode(cfg_v2()).unwrap();
    let layers = Decoder::default()
        .decode_all(Parser::default().parse_layers(&bytes).unwrap())
        .unwrap();
    let mlt_core::ParsedLayer::Tag02(layer) = &layers[0] else {
        panic!("expected a v2 layer")
    };
    let geometry = layer.layer().geometry_values();
    assert_eq!(geometry.vector_types(), [GeometryType::Mesh]);
    assert_eq!(geometry.vertex_count(0).unwrap(), 8);
    assert_eq!(geometry.mesh_triangles(0).unwrap().unwrap().len(), 12);
}
