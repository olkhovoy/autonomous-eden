use std::fs;
use std::path::Path;

use semiotic_hypercube::Node;

#[test]
fn roundtrip_sample_config() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../bin/Config/2018-10-29/2018-07-16 13-49-28 18 GRAMMAR PARAMETERS.34588-169384.cfg",
    );
    let data = fs::read(&path).expect("read sample config");
    let node = Node::from_bytes(&data).expect("parse sample config");
    let out = node.to_bytes().expect("serialize sample config");
    assert_eq!(data, out);
}
