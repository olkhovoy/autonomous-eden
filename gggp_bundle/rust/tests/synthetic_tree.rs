use semiotic_hypercube::Node;

#[test]
fn roundtrip_synthetic_tree() {
    let mut root = Node::new("root");
    root.set_global_id(42);
    root.set_int("answer", 42);
    root.set_real("pi", 3.1415);
    root.set_bool("flag", true);
    root.set_str("name", b"alpha".to_vec());
    root.set_time("when", 45123.75);
    root.set_sing("ratio", 1.25);

    root.set_int_array("ints", vec![1, 2, 3]);
    root.set_real_array("reals", vec![1.5, 2.5]);
    root.set_bool_array("bools", vec![true, false, true]);
    root.set_str_array(
        "strings",
        vec![b"a".to_vec(), b"bb".to_vec(), b"ccc".to_vec()],
    );
    root.set_time_array("times", vec![45123.0, 45124.5]);
    root.set_sing_array("sings", vec![0.5, 1.5]);
    root.set_bin("blob", vec![0x01, 0x02, 0x03]);
    root
        .set_bin_array(
            "records",
            2,
            3,
            vec![0x10, 0x11, 0x12, 0x20, 0x21, 0x22],
        )
        .expect("bin array set");

    let child = root.get_or_create_child("child");
    child.set_int("x", -7);

    let bytes = root.to_bytes().expect("serialize");
    let parsed = Node::from_bytes(&bytes).expect("parse");

    assert_eq!(parsed.get_int("answer"), 42);
    assert!((parsed.get_real("pi") - 3.1415).abs() < 1e-10);
    assert_eq!(parsed.get_bool("flag"), true);
    assert_eq!(parsed.get_str("name"), b"alpha".to_vec());
    assert!((parsed.get_time("when") - 45123.75).abs() < 1e-10);
    assert!((parsed.get_sing("ratio") - 1.25).abs() < 1e-6);

    assert_eq!(parsed.get_int_array("ints"), vec![1, 2, 3]);
    let reals = parsed.get_real_array("reals");
    assert_eq!(reals.len(), 2);
    assert!((reals[0] - 1.5).abs() < 1e-10);
    assert!((reals[1] - 2.5).abs() < 1e-10);
    assert_eq!(
        parsed.get_str_array("strings"),
        vec![b"a".to_vec(), b"bb".to_vec(), b"ccc".to_vec()]
    );

    assert_eq!(parsed.get_bin("blob").unwrap(), &[0x01, 0x02, 0x03]);
    let (count, record_size, data) = parsed.get_bin_array("records").unwrap();
    assert_eq!(count, 2);
    assert_eq!(record_size, 3);
    assert_eq!(data, &[0x10, 0x11, 0x12, 0x20, 0x21, 0x22]);

    assert_eq!(parsed.child("child").unwrap().get_int("x"), -7);
}
