use semiotic_hypercube::Node;
use semiotic_hypercube::gggp::parse_text;

fn build_test_grammar(dim: i32) -> Node {
    let mut root = Node::new("VECGRAM");
    root.set_int("MaxDepth", 24);
    root.set_int("MaxCrossoverNodes", 4);
    root.set_int("MaxMutationNodes", 4);

    let rules = root.get_or_create_child("RULES");

    let mut start = Node::new("START");
    let choices = start.get_or_create_child("CHOICES");
    let mut choice0 = Node::new("0");
    choice0.set_str("Text", b"<SEQ>".to_vec());
    choices.add_child(choice0);
    rules.add_child(start);

    let mut seq = Node::new("SEQ");
    let choices = seq.get_or_create_child("CHOICES");
    let mut c0 = Node::new("0");
    c0.set_str("Text", b"<OP> <SEQ>".to_vec());
    let mut c1 = Node::new("1");
    c1.set_str("Text", b"<OP>".to_vec());
    choices.add_child(c0);
    choices.add_child(c1);
    rules.add_child(seq);

    let mut op = Node::new("OP");
    let choices = op.get_or_create_child("CHOICES");
    let mut c0 = Node::new("0");
    let text_ax = format!("AX <axis from=0 to={} inc=1> <val from=-1 to=1 inc=0.5>", dim - 1);
    c0.set_str("Text", text_ax.as_bytes().to_vec());
    let mut c1 = Node::new("1");
    c1.set_str("Text", b"SCALE <val from=0.5 to=1.5 inc=0.5>".to_vec());
    let mut c2 = Node::new("2");
    c2.set_str("Text", b"NORM".to_vec());
    let mut c3 = Node::new("3");
    c3.set_str("Text", b"ZERO".to_vec());
    
    choices.add_child(c0);
    choices.add_child(c1);
    choices.add_child(c2);
    choices.add_child(c3);
    rules.add_child(op);

    root
}

fn main() {
    let mut grammar = build_test_grammar(3);
    if let Err(e) = grammar.to_file("../test_grammar.cfg") {
        eprintln!("Failed to write test_grammar.cfg: {}", e);
        std::process::exit(1);
    }
    println!("Generated test_grammar.cfg");
}
