//! One pass over a `CityGML` document by local names, so that `CityGML` 2.0 and 3.0 and any
//! namespace prefixes read alike.

use std::collections::HashMap;

use anyhow::{Result as AnyResult, bail};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

/// The part of a building a triangle belongs to, its `part` m-value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Wall,
    Roof,
    Ground,
    Window,
    Door,
}

impl Part {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::Roof => "roof",
            Self::Ground => "ground",
            Self::Window => "window",
            Self::Door => "door",
        }
    }
}

/// A polygon in source coordinates: its rings, the exterior first, none repeating its first
/// point at its end.
#[derive(Debug, PartialEq)]
pub struct Polygon {
    /// The semantic surface the polygon lies in, if any.
    pub part: Option<Part>,
    pub points: Vec<[f64; 3]>,
    /// Where each ring ends in `points`.
    pub ends: Vec<usize>,
}

/// A building, bridge or tunnel, with the polygons of the highest LOD of each of its parts.
#[derive(Debug, PartialEq)]
pub struct Building {
    pub gml_id: Option<String>,
    pub polygons: Vec<Polygon>,
}

/// What a document holds.
#[derive(Debug, Default, PartialEq)]
pub struct Document {
    /// The first `srsName` of the document.
    pub srs_name: Option<String>,
    pub buildings: Vec<Building>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tag {
    /// A city object that becomes one feature.
    Building,
    /// A part of a building that picks its own highest LOD, such as a building part or installation.
    Owner,
    Semantic(Part),
    /// Content that is not drawn from outside: closure surfaces, interiors, LOD4, appearances and
    /// implicit geometries.
    Skip,
    /// An `lodN…` geometry property.
    Lod(u8),
    Polygon,
    Interior,
    Ring,
    Coordinates,
    Other,
}

fn tag(name: &[u8]) -> Tag {
    match name {
        b"Building" | b"Bridge" | b"Tunnel" => Tag::Building,
        b"BuildingPart"
        | b"BuildingInstallation"
        | b"BuildingConstructiveElement"
        | b"BridgePart"
        | b"BridgeInstallation"
        | b"BridgeConstructionElement"
        | b"BridgeConstructiveElement"
        | b"TunnelPart"
        | b"TunnelInstallation"
        | b"TunnelConstructiveElement" => Tag::Owner,
        b"WallSurface" | b"OuterCeilingSurface" => Tag::Semantic(Part::Wall),
        b"RoofSurface" | b"OuterFloorSurface" => Tag::Semantic(Part::Roof),
        b"GroundSurface" => Tag::Semantic(Part::Ground),
        b"Window" | b"WindowSurface" => Tag::Semantic(Part::Window),
        b"Door" | b"DoorSurface" => Tag::Semantic(Part::Door),
        b"ClosureSurface"
        | b"InteriorWallSurface"
        | b"CeilingSurface"
        | b"FloorSurface"
        | b"interiorRoom"
        | b"interiorBuildingInstallation"
        | b"interiorFurniture"
        | b"interiorBridgeRoom"
        | b"interiorBridgeInstallation"
        | b"interiorTunnelInstallation"
        | b"interiorHollowSpace"
        | b"buildingRoom"
        | b"buildingFurniture"
        | b"bridgeRoom"
        | b"bridgeFurniture"
        | b"hollowSpace"
        | b"tunnelFurniture"
        | b"appearance"
        | b"appearanceMember"
        | b"ImplicitGeometry" => Tag::Skip,
        b"Polygon" | b"Triangle" | b"Rectangle" | b"PolygonPatch" => Tag::Polygon,
        b"interior" | b"innerBoundaryIs" => Tag::Interior,
        b"LinearRing" => Tag::Ring,
        b"posList" | b"pos" | b"coordinates" => Tag::Coordinates,
        _ => lod_tag(name).unwrap_or(Tag::Other),
    }
}

/// `lodN…` geometry properties: LOD4 (interiors), curves and terrain intersections are skipped.
fn lod_tag(name: &[u8]) -> Option<Tag> {
    let rest = name.strip_prefix(b"lod")?;
    let (&digit, kind) = rest.split_first()?;
    if !digit.is_ascii_digit() || !kind.first()?.is_ascii_uppercase() {
        return None;
    }
    if digit == b'4' || kind.ends_with(b"MultiCurve") || kind.starts_with(b"TerrainIntersection") {
        return Some(Tag::Skip);
    }
    Some(Tag::Lod(digit - b'0'))
}

/// Reads the buildings of `bytes`.
pub fn parse(bytes: &[u8]) -> AnyResult<Document> {
    let mut parser = Parser::default();
    let mut reader = Reader::from_reader(bytes);
    reader.config_mut().check_end_names = false;
    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => parser.start(&e),
            Ok(Event::Empty(e)) => {
                parser.start(&e);
                parser.end();
            }
            Ok(Event::End(_)) => parser.end(),
            Ok(Event::Text(t)) if parser.in_coordinates => {
                parse_numbers(t.as_bytes(), &mut parser.numbers);
            }
            Ok(Event::Eof) => return Ok(parser.document),
            Ok(_) => {}
            Err(e) => bail!("invalid XML at byte {}: {e}", reader.buffer_position()),
        }
    }
}

#[derive(Default)]
struct Parser {
    stack: Vec<Tag>,
    /// The depth of a skipped element: everything inside it is left unread.
    skip: Option<usize>,
    /// The building being read, its polygons collected in `polygons` until it ends.
    building: Option<Building>,
    /// The polygons of the building, each with its owner and LOD.
    polygons: Vec<(u32, u8, Polygon)>,
    owners: Vec<u32>,
    next_owner: u32,
    lods: Vec<u8>,
    /// The semantic surface of the innermost surface or owner, `None` for an owner.
    semantics: Vec<Option<Part>>,
    in_coordinates: bool,
    two_d: bool,
    numbers: Vec<f64>,
    points: Vec<[f64; 3]>,
    ends: Vec<usize>,
    polygon_ok: bool,
    document: Document,
}

fn attribute(e: &BytesStart<'_>, local: &[u8]) -> Option<String> {
    e.attributes()
        .flatten()
        .find(|a| a.key.local_name().as_ref().as_bytes() == local)
        .map(|a| a.value.into_owned())
}

impl Parser {
    fn start(&mut self, e: &BytesStart<'_>) {
        if self.document.srs_name.is_none() {
            self.document.srs_name = attribute(e, b"srsName");
        }
        let depth = self.stack.len();
        let tag = match (self.skip, tag(e.local_name().as_ref().as_bytes())) {
            (Some(_), _) => Tag::Other,
            (None, Tag::Building) if self.building.is_some() => Tag::Owner,
            (None, tag) => tag,
        };
        #[expect(
            clippy::wildcard_enum_match_arm,
            reason = "only the tags that collect something start anything"
        )]
        match tag {
            Tag::Building => {
                self.building = Some(Building {
                    gml_id: attribute(e, b"id"),
                    polygons: Vec::new(),
                });
                self.polygons.clear();
            }
            Tag::Skip => self.skip = Some(depth),
            Tag::Owner => {
                self.next_owner += 1;
                self.owners.push(self.next_owner);
                self.semantics.push(None);
            }
            Tag::Semantic(part) => self.semantics.push(Some(part)),
            Tag::Lod(lod) => self.lods.push(lod),
            Tag::Polygon => {
                self.points = Vec::new();
                self.ends = Vec::new();
                self.polygon_ok = true;
            }
            Tag::Ring => {
                self.numbers.clear();
                self.two_d = false;
            }
            Tag::Coordinates => {
                self.in_coordinates = true;
                if attribute(e, b"srsDimension").is_some_and(|d| d.trim() != "3") {
                    self.two_d = true;
                }
            }
            _ => {}
        }
        self.stack.push(tag);
    }

    fn end(&mut self) {
        let Some(tag) = self.stack.pop() else {
            return;
        };
        match tag {
            Tag::Building => self.end_building(),
            Tag::Skip => self.skip = None,
            Tag::Owner => {
                self.owners.pop();
                self.semantics.pop();
            }
            Tag::Semantic(_) => {
                self.semantics.pop();
            }
            Tag::Lod(_) => {
                self.lods.pop();
            }
            Tag::Coordinates => self.in_coordinates = false,
            Tag::Ring => self.end_ring(),
            Tag::Polygon => self.end_polygon(),
            Tag::Interior | Tag::Other => {}
        }
    }

    fn end_ring(&mut self) {
        let interior = self.stack.last() == Some(&Tag::Interior);
        let start = self.points.len();
        if !self.two_d && self.numbers.len().is_multiple_of(3) {
            for c in self.numbers.as_chunks::<3>().0 {
                let p = [c[0], c[1], c[2]];
                if self.points.len() == start || self.points.last() != Some(&p) {
                    self.points.push(p);
                }
            }
            if self.points.len() > start + 1 && self.points.last() == self.points.get(start) {
                self.points.pop();
            }
        }
        if self.points.len() < start + 3 {
            self.points.truncate(start);
            if !interior {
                self.polygon_ok = false;
            }
            return;
        }
        // Only the first exterior ring and the holes after it count.
        if interior == self.ends.is_empty() {
            self.points.truncate(start);
            return;
        }
        self.ends.push(self.points.len());
    }

    fn end_polygon(&mut self) {
        let Some(&lod) = self.lods.last() else {
            return;
        };
        if self.building.is_none() || !self.polygon_ok || self.ends.is_empty() {
            return;
        }
        let polygon = Polygon {
            part: self.semantics.last().copied().flatten(),
            points: std::mem::take(&mut self.points),
            ends: std::mem::take(&mut self.ends),
        };
        let owner = self.owners.last().copied().unwrap_or(0);
        self.polygons.push((owner, lod, polygon));
    }

    fn end_building(&mut self) {
        let Some(mut building) = self.building.take() else {
            return;
        };
        let mut best: HashMap<u32, u8> = HashMap::new();
        for &(owner, lod, _) in &self.polygons {
            let best = best.entry(owner).or_insert(lod);
            *best = (*best).max(lod);
        }
        building.polygons = self
            .polygons
            .drain(..)
            .filter(|(owner, lod, _)| best.get(owner) == Some(lod))
            .map(|(_, _, polygon)| polygon)
            .collect();
        if !building.polygons.is_empty() {
            self.document.buildings.push(building);
        }
    }
}

/// Appends the numbers of `text`, separated by whitespace or commas, to `out`; others are skipped.
fn parse_numbers(text: &[u8], out: &mut Vec<f64>) {
    let Ok(text) = std::str::from_utf8(text) else {
        return;
    };
    out.extend(
        text.split(|c: char| c.is_ascii_whitespace() || c == ',')
            .filter_map(|s| s.parse::<f64>().ok()),
    );
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    const HEAD: &str = r#"<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0"
        xmlns:bldg="http://www.opengis.net/citygml/building/2.0" xmlns:gml="http://www.opengis.net/gml">"#;

    fn ring(points: &[[f64; 3]]) -> String {
        let list: Vec<String> = points
            .iter()
            .chain(&points[..1])
            .map(|p| format!("{} {} {}", p[0], p[1], p[2]))
            .collect();
        format!(
            "<gml:LinearRing><gml:posList>{}</gml:posList></gml:LinearRing>",
            list.join(" ")
        )
    }

    fn polygon(rings: &[&[[f64; 3]]]) -> String {
        let mut out = format!(
            "<gml:Polygon><gml:exterior>{}</gml:exterior>",
            ring(rings[0])
        );
        for hole in &rings[1..] {
            write!(out, "<gml:interior>{}</gml:interior>", ring(hole)).unwrap();
        }
        out + "</gml:Polygon>"
    }

    fn surface(semantic: &str, lod: u8, polygon: &str) -> String {
        format!(
            "<bldg:boundedBy><bldg:{semantic}><bldg:lod{lod}MultiSurface><gml:MultiSurface>\
             <gml:surfaceMember>{polygon}</gml:surfaceMember></gml:MultiSurface>\
             </bldg:lod{lod}MultiSurface></bldg:{semantic}></bldg:boundedBy>"
        )
    }

    const SQUARE: [[f64; 3]; 4] = [
        [0.0, 0.0, 10.0],
        [10.0, 0.0, 10.0],
        [10.0, 10.0, 10.0],
        [0.0, 10.0, 10.0],
    ];
    const HOLE: [[f64; 3]; 3] = [[2.0, 2.0, 10.0], [4.0, 2.0, 10.0], [2.0, 4.0, 10.0]];

    #[test]
    fn a_building_keeps_its_highest_lod_with_semantics_and_holes() {
        let xml = format!(
            r#"{HEAD}<gml:Envelope srsName="EPSG:25832"/><core:cityObjectMember><bldg:Building gml:id="B1">{}{}{}</bldg:Building></core:cityObjectMember></core:CityModel>"#,
            surface("RoofSurface", 1, &polygon(&[&SQUARE])),
            surface("RoofSurface", 2, &polygon(&[&SQUARE, &HOLE])),
            surface("ClosureSurface", 2, &polygon(&[&SQUARE])),
        );
        let document = parse(xml.as_bytes()).unwrap();
        assert_eq!(document.srs_name.as_deref(), Some("EPSG:25832"));
        let [building] = &document.buildings[..] else {
            panic!("{document:?}");
        };
        assert_eq!(building.gml_id.as_deref(), Some("B1"));
        let [roof] = &building.polygons[..] else {
            panic!("{building:?}");
        };
        assert_eq!(roof.part, Some(Part::Roof));
        assert_eq!(roof.ends, [4, 7]);
        assert_eq!(roof.points[4..], HOLE);
    }

    #[test]
    fn each_part_of_a_building_picks_its_own_lod() {
        let xml = format!(
            "{HEAD}<bldg:Building>{}<bldg:consistsOfBuildingPart><bldg:BuildingPart>{}\
             </bldg:BuildingPart></bldg:consistsOfBuildingPart></bldg:Building></core:CityModel>",
            surface("WallSurface", 3, &polygon(&[&SQUARE])),
            surface("GroundSurface", 2, &polygon(&[&SQUARE])),
        );
        let document = parse(xml.as_bytes()).unwrap();
        let parts: Vec<_> = document.buildings[0]
            .polygons
            .iter()
            .map(|p| p.part)
            .collect();
        assert_eq!(parts, [Some(Part::Wall), Some(Part::Ground)]);
        assert_eq!(document.buildings[0].gml_id, None);
    }

    #[test]
    fn rings_without_area_and_2d_rings_are_left_out() {
        let line = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let flat = "<gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension=\"2\">\
                    0 0 1 0 1 1 0 0</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>";
        let xml = format!(
            "{HEAD}<bldg:Building>{}{}</bldg:Building></core:CityModel>",
            surface("WallSurface", 2, &polygon(&[&line])),
            surface("WallSurface", 2, flat),
        );
        assert_eq!(parse(xml.as_bytes()).unwrap().buildings, []);
    }

    #[test]
    fn numbers_are_split_by_whitespace_and_commas() {
        let mut out = Vec::new();
        parse_numbers(b" 1.5,2\n-3e2 x 4", &mut out);
        assert_eq!(out, [1.5, 2.0, -300.0, 4.0]);
    }

    #[test]
    fn broken_xml_fails_with_its_position() {
        let err = parse(b"<a><b></a").unwrap_err();
        assert!(err.to_string().starts_with("invalid XML at byte"), "{err}");
    }
}
