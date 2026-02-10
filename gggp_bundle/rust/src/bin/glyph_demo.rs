use std::fs::File;
use std::io::{self, Write};

use gor_data_storage::gggp::{calc_lengths, parse_text, Gggp, GpIndividual};
use gor_data_storage::Node;

#[derive(Clone, Copy)]
struct Segment {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
}

#[derive(Clone)]
struct TargetGlyph {
    name: &'static str,
    segments: Vec<Segment>,
    bitmap: Vec<u8>,
}

#[derive(Clone)]
struct Glyph {
    segments: Vec<Segment>,
}

struct Args {
    gens: usize,
    pop: usize,
    elite: usize,
    out: String,
    grid: usize,
    letters: usize,
}

fn main() -> io::Result<()> {
    let args = parse_args();

    let mut cfg = build_grammar();
    finalize_grammar(&mut cfg);

    let targets_fit = build_targets(args.grid, args.letters);
    let cfgs: Vec<Node> = (0..targets_fit.len()).map(|_| cfg.clone()).collect();

    let mut gggp = Gggp::new();
    gggp.set_on_get_fitness(move |ind| Some(score_individual(ind, &targets_fit, args.grid)));
    gggp
        .init_from_nodes(&cfgs, args.pop, args.elite, 0.7, 0.3)
        .expect("init gggp failed");

    let targets_score = build_targets(args.grid, args.letters);
    let mut best_score = f64::MIN;
    let mut best_ind: Option<GpIndividual> = None;

    for gen in 0..args.gens {
        gggp.step();
        for ind in gggp.individuals() {
            let score = score_individual(ind, &targets_score, args.grid);
            if score > best_score {
                best_score = score;
                best_ind = Some(ind.clone());
            }
        }
        if gen % 10 == 0 || gen + 1 == args.gens {
            println!("gen {} best {:.6}", gen + 1, best_score);
        }
    }

    let best_ind = best_ind.expect("no best individual");
    let targets = build_targets(args.grid, args.letters);
    let glyphs = build_glyphs(&best_ind, targets.len());
    write_svg(&args.out, &glyphs, &targets, 220, 20)?;
    println!("saved {}", args.out);
    Ok(())
}

fn parse_args() -> Args {
    let mut args = Args {
        gens: 60,
        pop: 80,
        elite: 6,
        out: "glyph_demo.svg".to_string(),
        grid: 32,
        letters: 6,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--gens" => args.gens = read_usize(&mut iter, "--gens"),
            "--pop" => args.pop = read_usize(&mut iter, "--pop"),
            "--elite" => args.elite = read_usize(&mut iter, "--elite"),
            "--out" => args.out = read_string(&mut iter, "--out"),
            "--grid" => args.grid = read_usize(&mut iter, "--grid"),
            "--letters" => args.letters = read_usize(&mut iter, "--letters"),
            _ => {}
        }
    }

    args
}

fn read_usize(iter: &mut impl Iterator<Item = String>, flag: &str) -> usize {
    iter.next()
        .unwrap_or_else(|| panic!("missing value for {}", flag))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid value for {}", flag))
}

fn read_string(iter: &mut impl Iterator<Item = String>, flag: &str) -> String {
    iter.next().unwrap_or_else(|| panic!("missing value for {}", flag))
}

fn build_grammar() -> Node {
    let mut root = Node::new("GLYPH");
    root.set_int("MaxDepth", 6);
    root.set_int("MaxCrossoverNodes", 4);
    root.set_int("MaxMutationNodes", 4);

    let rules = root.get_or_create_child("RULES");

    let mut start = Node::new("START");
    let choices = start.get_or_create_child("CHOICES");
    add_choice(choices, 0, "<SEQ>");
    rules.add_child(start);

    let mut seq = Node::new("SEQ");
    let choices = seq.get_or_create_child("CHOICES");
    add_choice(choices, 0, "<SEG> <SEQ>");
    add_choice(choices, 1, "<SEG>");
    rules.add_child(seq);

    let mut seg = Node::new("SEG");
    let choices = seg.get_or_create_child("CHOICES");
    add_choice(choices, 0, "F <len from=0.2 to=1 inc=0.2>");
    add_choice(choices, 1, "L <ang from=15 to=180 inc=15>");
    add_choice(choices, 2, "R <ang from=15 to=180 inc=15>");
    add_choice(choices, 3, "Z");
    rules.add_child(seg);

    root
}

fn add_choice(choices: &mut Node, index: i32, text: &str) {
    let mut choice = Node::new(index.to_string());
    choice.set_str("Text", text);
    choices.add_child(choice);
}

fn finalize_grammar(cfg: &mut Node) {
    if let Some(rules) = cfg.child_mut("RULES") {
        for symbol in rules.children_mut() {
            if let Some(choices) = symbol.child_mut("CHOICES") {
                for choice in choices.children_mut() {
                    parse_text(choice);
                }
            }
        }
    }
    calc_lengths(cfg).expect("calc_lengths failed");
}

fn build_targets(grid: usize, limit: usize) -> Vec<TargetGlyph> {
    let mut targets = Vec::new();

    let cross = vec![
        Segment {
            x1: -0.6,
            y1: 0.0,
            x2: 0.6,
            y2: 0.0,
        },
        Segment {
            x1: 0.0,
            y1: -0.6,
            x2: 0.0,
            y2: 0.6,
        },
    ];
    targets.push(make_target("AZ", cross, grid));

    let mut buki = Vec::new();
    buki.push(Segment {
        x1: -0.5,
        y1: -0.6,
        x2: -0.5,
        y2: 0.6,
    });
    buki.extend(rect_segments(0.1, 0.3, 0.6, 0.45));
    buki.extend(rect_segments(0.1, -0.3, 0.6, 0.45));
    targets.push(make_target("BUKI", buki, grid));

    let vedi = triangle_segments();
    targets.push(make_target("VEDI", vedi, grid));

    let mut glagol = rect_segments(0.0, 0.0, 0.8, 0.8);
    glagol.push(Segment {
        x1: 0.0,
        y1: -0.6,
        x2: 0.0,
        y2: 0.6,
    });
    targets.push(make_target("GLAGOL", glagol, grid));

    let dobro = rect_segments(0.0, 0.0, 0.7, 0.9);
    targets.push(make_target("DOBRO", dobro, grid));

    let est = vec![
        Segment {
            x1: -0.6,
            y1: 0.5,
            x2: 0.6,
            y2: 0.5,
        },
        Segment {
            x1: -0.6,
            y1: 0.0,
            x2: 0.6,
            y2: 0.0,
        },
        Segment {
            x1: -0.6,
            y1: -0.5,
            x2: 0.6,
            y2: -0.5,
        },
    ];
    targets.push(make_target("EST", est, grid));

    let hourglass = vec![
        Segment {
            x1: -0.6,
            y1: 0.6,
            x2: 0.6,
            y2: -0.6,
        },
        Segment {
            x1: -0.6,
            y1: -0.6,
            x2: 0.6,
            y2: 0.6,
        },
    ];
    targets.push(make_target("IZHE", hourglass, grid));

    let mut slovo = rect_segments(0.0, 0.0, 0.8, 0.8);
    slovo.extend(triangle_segments());
    targets.push(make_target("SLOVO", slovo, grid));

    let mut on = rect_segments(-0.35, 0.0, 0.45, 0.6);
    on.extend(rect_segments(0.35, 0.0, 0.45, 0.6));
    targets.push(make_target("ON", on, grid));

    if limit > 0 && targets.len() > limit {
        targets.truncate(limit);
    }
    targets
}

fn triangle_segments() -> Vec<Segment> {
    let p1 = (0.0, 0.6);
    let p2 = (-0.5, -0.3);
    let p3 = (0.5, -0.3);
    vec![
        Segment {
            x1: p1.0,
            y1: p1.1,
            x2: p2.0,
            y2: p2.1,
        },
        Segment {
            x1: p2.0,
            y1: p2.1,
            x2: p3.0,
            y2: p3.1,
        },
        Segment {
            x1: p3.0,
            y1: p3.1,
            x2: p1.0,
            y2: p1.1,
        },
    ]
}

fn rect_segments(cx: f64, cy: f64, w: f64, h: f64) -> Vec<Segment> {
    let hw = w * 0.5;
    let hh = h * 0.5;
    let x1 = cx - hw;
    let x2 = cx + hw;
    let y1 = cy - hh;
    let y2 = cy + hh;
    vec![
        Segment {
            x1,
            y1,
            x2,
            y2: y1,
        },
        Segment {
            x1: x2,
            y1,
            x2,
            y2,
        },
        Segment {
            x1: x2,
            y1: y2,
            x2: x1,
            y2,
        },
        Segment {
            x1,
            y1: y2,
            x2: x1,
            y2: y1,
        },
    ]
}

fn make_target(name: &'static str, segments: Vec<Segment>, grid: usize) -> TargetGlyph {
    let bitmap = rasterize(&segments, grid);
    TargetGlyph {
        name,
        segments,
        bitmap,
    }
}

fn build_glyphs(ind: &GpIndividual, count: usize) -> Vec<Glyph> {
    let mut glyphs = Vec::new();
    for (i, tree) in ind.trees().iter().enumerate() {
        if i >= count {
            break;
        }
        let text = tree.text();
        let glyph = glyph_from_text(&text);
        glyphs.push(glyph);
    }
    glyphs
}

fn score_individual(ind: &GpIndividual, targets: &[TargetGlyph], grid: usize) -> f64 {
    let glyphs = build_glyphs(ind, targets.len());
    if glyphs.len() != targets.len() {
        return 0.0;
    }

    let mut scores = Vec::new();
    let mut bitmaps = Vec::new();

    for (glyph, target) in glyphs.iter().zip(targets.iter()) {
        let score = score_glyph(glyph, target, grid);
        scores.push(score);
        bitmaps.push(rasterize(&glyph.segments, grid));
    }

    let avg_score = scores.iter().copied().sum::<f64>() / scores.len() as f64;
    let distinct = distinctness(&bitmaps);

    avg_score + 0.2 * distinct
}

fn score_glyph(glyph: &Glyph, target: &TargetGlyph, grid: usize) -> f64 {
    if glyph.segments.is_empty() {
        return 0.0;
    }
    let bitmap = rasterize(&glyph.segments, grid);
    let sim = f1_similarity(&bitmap, &target.bitmap);
    let closure = closure_score(&glyph.segments);
    let complexity = complexity_score(glyph.segments.len());
    let symmetry = symmetry_score(&bitmap, grid);

    0.45 * sim + 0.3 * closure + 0.15 * complexity + 0.1 * symmetry
}

fn closure_score(segments: &[Segment]) -> f64 {
    if segments.is_empty() {
        return 0.0;
    }
    let first = segments.first().unwrap();
    let last = segments.last().unwrap();
    let dx = last.x2 - first.x1;
    let dy = last.y2 - first.y1;
    let dist = (dx * dx + dy * dy).sqrt();
    let (min_x, min_y, max_x, max_y) = bounds(segments);
    let scale = (max_x - min_x).max(max_y - min_y);
    if scale <= 1e-6 {
        return 0.0;
    }
    let ratio = (dist / scale).min(1.0);
    1.0 - ratio
}

fn symmetry_score(bitmap: &[u8], grid: usize) -> f64 {
    if grid == 0 || bitmap.len() != grid * grid {
        return 0.0;
    }
    let mut vert_match = 0u32;
    let mut vert_pairs = 0u32;
    let mut horiz_match = 0u32;
    let mut horiz_pairs = 0u32;

    for y in 0..grid {
        for x in 0..(grid / 2) {
            let left = bitmap[y * grid + x];
            let right = bitmap[y * grid + (grid - 1 - x)];
            if left == right {
                vert_match += 1;
            }
            vert_pairs += 1;
        }
    }

    for y in 0..(grid / 2) {
        for x in 0..grid {
            let top = bitmap[y * grid + x];
            let bottom = bitmap[(grid - 1 - y) * grid + x];
            if top == bottom {
                horiz_match += 1;
            }
            horiz_pairs += 1;
        }
    }

    let vert = if vert_pairs == 0 {
        0.0
    } else {
        vert_match as f64 / vert_pairs as f64
    };
    let horiz = if horiz_pairs == 0 {
        0.0
    } else {
        horiz_match as f64 / horiz_pairs as f64
    };
    0.5 * (vert + horiz)
}

fn complexity_score(count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let max_count = 40.0;
    let ratio = (count as f64 / max_count).min(1.0);
    1.0 - ratio
}

fn distinctness(bitmaps: &[Vec<u8>]) -> f64 {
    if bitmaps.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut pairs = 0.0;
    for i in 0..bitmaps.len() {
        for j in (i + 1)..bitmaps.len() {
            total += hamming_distance(&bitmaps[i], &bitmaps[j]);
            pairs += 1.0;
        }
    }
    if pairs == 0.0 {
        0.0
    } else {
        total / pairs
    }
}

fn hamming_distance(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut diff = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x != y {
            diff += 1;
        }
    }
    diff as f64 / a.len() as f64
}

fn f1_similarity(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut inter = 0u32;
    let mut suma = 0u32;
    let mut sumb = 0u32;
    for (x, y) in a.iter().zip(b.iter()) {
        if *x != 0 {
            suma += 1;
        }
        if *y != 0 {
            sumb += 1;
        }
        if *x != 0 && *y != 0 {
            inter += 1;
        }
    }
    if suma == 0 || sumb == 0 {
        return 0.0;
    }
    let precision = inter as f64 / suma as f64;
    let recall = inter as f64 / sumb as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn glyph_from_text(text: &str) -> Glyph {
    let mut segments = Vec::new();
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0usize;

    let mut x = 0.0;
    let mut y = 0.0;
    let mut heading = 0.0f64;
    let mut start = Some((x, y));
    let mut segment_limit = 60usize;

    while i < tokens.len() && segment_limit > 0 {
        let token = tokens[i];
        match token {
            "F" => {
                if i + 1 < tokens.len() {
                    if let Some(len) = parse_number(tokens[i + 1]) {
                        let nx = x + len * heading.cos();
                        let ny = y + len * heading.sin();
                        segments.push(Segment { x1: x, y1: y, x2: nx, y2: ny });
                        x = nx;
                        y = ny;
                        if start.is_none() {
                            start = Some((x, y));
                        }
                        segment_limit -= 1;
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "L" => {
                if i + 1 < tokens.len() {
                    if let Some(angle) = parse_number(tokens[i + 1]) {
                        heading += angle.to_radians();
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "R" => {
                if i + 1 < tokens.len() {
                    if let Some(angle) = parse_number(tokens[i + 1]) {
                        heading -= angle.to_radians();
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "Z" => {
                if let Some((sx, sy)) = start {
                    segments.push(Segment { x1: x, y1: y, x2: sx, y2: sy });
                    x = sx;
                    y = sy;
                    segment_limit = segment_limit.saturating_sub(1);
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    Glyph { segments }
}

fn parse_number(token: &str) -> Option<f64> {
    let mut end = 0usize;
    for (idx, ch) in token.char_indices() {
        if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E' {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    token[..end].parse::<f64>().ok()
}

fn rasterize(segments: &[Segment], grid: usize) -> Vec<u8> {
    let mut bitmap = vec![0u8; grid * grid];
    if segments.is_empty() || grid == 0 {
        return bitmap;
    }
    let (min_x, min_y, max_x, max_y) = bounds(segments);
    let scale = (max_x - min_x).max(max_y - min_y);
    if scale <= 1e-6 {
        return bitmap;
    }

    for seg in segments {
        let dx = seg.x2 - seg.x1;
        let dy = seg.y2 - seg.y1;
        let length = (dx * dx + dy * dy).sqrt();
        let steps = ((length * 30.0).ceil() as usize).max(2);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let x = seg.x1 + dx * t;
            let y = seg.y1 + dy * t;
            let nx = (x - min_x) / scale;
            let ny = (y - min_y) / scale;
            let ix = (nx * (grid as f64 - 1.0)).round() as isize;
            let iy = (ny * (grid as f64 - 1.0)).round() as isize;
            if ix >= 0 && iy >= 0 && (ix as usize) < grid && (iy as usize) < grid {
                bitmap[iy as usize * grid + ix as usize] = 1;
            }
        }
    }

    bitmap
}

fn bounds(segments: &[Segment]) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for seg in segments {
        min_x = min_x.min(seg.x1).min(seg.x2);
        min_y = min_y.min(seg.y1).min(seg.y2);
        max_x = max_x.max(seg.x1).max(seg.x2);
        max_y = max_y.max(seg.y1).max(seg.y2);
    }

    (min_x, min_y, max_x, max_y)
}

fn write_svg(path: &str, glyphs: &[Glyph], targets: &[TargetGlyph], cell: i32, pad: i32) -> io::Result<()> {
    let cols = 2;
    let rows = ((glyphs.len() + cols - 1) / cols).max(1);
    let width = cols as i32 * cell + (cols as i32 + 1) * pad;
    let height = rows as i32 * cell + (rows as i32 + 1) * pad;

    let mut file = File::create(path)?;
    writeln!(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">", width, height)?;
    writeln!(file, "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>")?;

    for (idx, glyph) in glyphs.iter().enumerate() {
        let row = idx / cols;
        let col = idx % cols;
        let ox = pad + col as i32 * (cell + pad);
        let oy = pad + row as i32 * (cell + pad);

        let target = targets.get(idx);
        if let Some(target) = target {
            let tgt_path = segments_to_path(&normalize_segments(&target.segments, cell, 10), ox, oy);
            writeln!(file, "<path d=\"{}\" fill=\"none\" stroke=\"#cccccc\" stroke-width=\"2\"/>", tgt_path)?;
            writeln!(file, "<text x=\"{}\" y=\"{}\" font-size=\"12\" fill=\"#666\">{}</text>", ox, oy - 4, target.name)?;
        }

        let path_data = segments_to_path(&normalize_segments(&glyph.segments, cell, 10), ox, oy);
        writeln!(file, "<path d=\"{}\" fill=\"none\" stroke=\"#111111\" stroke-width=\"2\"/>", path_data)?;
    }

    writeln!(file, "</svg>")?;
    Ok(())
}

fn normalize_segments(segments: &[Segment], size: i32, padding: i32) -> Vec<Segment> {
    if segments.is_empty() {
        return Vec::new();
    }
    let (min_x, min_y, max_x, max_y) = bounds(segments);
    let scale = (max_x - min_x).max(max_y - min_y);
    if scale <= 1e-6 {
        return Vec::new();
    }

    let usable = (size - 2 * padding) as f64;
    let mut out = Vec::new();
    for seg in segments {
        let x1 = (seg.x1 - min_x) / scale;
        let y1 = (seg.y1 - min_y) / scale;
        let x2 = (seg.x2 - min_x) / scale;
        let y2 = (seg.y2 - min_y) / scale;

        let sx1 = padding as f64 + x1 * usable;
        let sy1 = padding as f64 + (1.0 - y1) * usable;
        let sx2 = padding as f64 + x2 * usable;
        let sy2 = padding as f64 + (1.0 - y2) * usable;

        out.push(Segment {
            x1: sx1,
            y1: sy1,
            x2: sx2,
            y2: sy2,
        });
    }
    out
}

fn segments_to_path(segments: &[Segment], ox: i32, oy: i32) -> String {
    let mut path = String::new();
    for seg in segments {
        let x1 = seg.x1 + ox as f64;
        let y1 = seg.y1 + oy as f64;
        let x2 = seg.x2 + ox as f64;
        let y2 = seg.y2 + oy as f64;
        path.push_str(&format!("M {:.2} {:.2} L {:.2} {:.2} ", x1, y1, x2, y2));
    }
    path
}
