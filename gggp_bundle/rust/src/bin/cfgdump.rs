use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use chrono::{Duration, NaiveDate, NaiveDateTime};
use gor_data_storage::{DataType, Node, Value};

const ARRAY_PREVIEW_LIMIT: usize = 32;
const BIN_PREVIEW_BYTES: usize = 32;

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Txt,
    Md,
    Html,
}

struct Config {
    input: String,
    format: OutputFormat,
    output: Option<String>,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{}", usage());
        return;
    }

    let config = match parse_args(args.into_iter()) {
        Ok(config) => config,
        Err(msg) => {
            eprintln!("{msg}");
            eprint!("{}", usage());
            std::process::exit(2);
        }
    };

    let node = match load_node(&config.input) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("error reading {}: {err}", config.input);
            std::process::exit(1);
        }
    };

    let title = title_from_input(&config.input);
    let rendered = match config.format {
        OutputFormat::Txt => render_txt(&node),
        OutputFormat::Md => render_md(&node),
        OutputFormat::Html => render_html(&node, &title),
    };

    if let Err(err) = write_output(config.output.as_deref(), rendered.as_bytes()) {
        eprintln!("error writing output: {err}");
        std::process::exit(1);
    }
}

fn usage() -> &'static str {
    "Usage: cfgdump [-f txt|md|html] [-o OUTPUT] <input.cfg|->\n\
\n\
Options:\n\
  -f, --format  Output format: txt (default), md, html\n\
  -o, --output  Output file (default: stdout)\n\
  -h, --help    Show this help message\n\
"
}

fn parse_args<I>(mut args: I) -> Result<Config, String>
where
    I: Iterator<Item = String>,
{
    let mut format = OutputFormat::Txt;
    let mut output = None;
    let mut input = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-f" | "--format" => {
                let value = args.next().ok_or("missing value for -f")?;
                format = parse_format(&value)?;
            }
            "-o" | "--output" => {
                let value = args.next().ok_or("missing value for -o")?;
                output = Some(value);
            }
            _ if arg.starts_with("-f=") => {
                let value = &arg[3..];
                format = parse_format(value)?;
            }
            _ if arg.starts_with("--format=") => {
                let value = &arg["--format=".len()..];
                format = parse_format(value)?;
            }
            _ if arg.starts_with("-o=") => {
                let value = &arg[3..];
                output = Some(value.to_string());
            }
            _ if arg.starts_with("--output=") => {
                let value = &arg["--output=".len()..];
                output = Some(value.to_string());
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option: {arg}"));
            }
            _ => {
                if input.is_some() {
                    return Err("multiple input files provided".to_string());
                }
                input = Some(arg);
            }
        }
    }

    let input = input.ok_or("missing input file".to_string())?;
    Ok(Config {
        input,
        format,
        output,
    })
}

fn parse_format(value: &str) -> Result<OutputFormat, String> {
    match value.to_ascii_lowercase().as_str() {
        "txt" | "text" => Ok(OutputFormat::Txt),
        "md" | "markdown" => Ok(OutputFormat::Md),
        "html" | "htm" => Ok(OutputFormat::Html),
        _ => Err(format!("unknown format: {value}")),
    }
}

fn load_node(input: &str) -> Result<Node, String> {
    if input == "-" {
        let mut stdin = io::stdin().lock();
        Node::from_reader(&mut stdin).map_err(|err| err.to_string())
    } else {
        Node::from_file(input).map_err(|err| err.to_string())
    }
}

fn title_from_input(input: &str) -> String {
    if input == "-" {
        "cfgdump".to_string()
    } else {
        Path::new(input)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("cfgdump")
            .to_string()
    }
}

fn write_output(output: Option<&str>, data: &[u8]) -> io::Result<()> {
    match output {
        Some(path) => {
            let mut file = File::create(path)?;
            file.write_all(data)
        }
        None => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(data)
        }
    }
}

fn render_txt(node: &Node) -> String {
    let mut out = String::new();
    write_txt_node(node, &mut out, 0);
    out
}

fn write_txt_node(node: &Node, out: &mut String, indent: usize) {
    push_indent(out, indent);
    out.push_str(&format_name(node.name()));
    out.push('\n');

    let child_indent = indent + 2;
    for attr in node.attrs() {
        push_indent(out, child_indent);
        out.push('@');
        out.push_str(&format_name(attr.name()));
        out.push_str(" (");
        out.push_str(data_type_name(attr.value().data_type()));
        out.push_str(") = ");
        out.push_str(&format_value(attr.value()));
        out.push('\n');
    }

    for child in node.children() {
        write_txt_node(child, out, child_indent);
    }
}

fn render_md(node: &Node) -> String {
    let mut out = String::new();
    write_md_node(node, &mut out, 0);
    out
}

fn write_md_node(node: &Node, out: &mut String, indent: usize) {
    push_indent(out, indent);
    out.push_str("- ");
    out.push_str(&format_name(node.name()));
    out.push('\n');

    let child_indent = indent + 2;
    for attr in node.attrs() {
        push_indent(out, child_indent);
        out.push_str("- @");
        out.push_str(&format_name(attr.name()));
        out.push_str(" (");
        out.push_str(data_type_name(attr.value().data_type()));
        out.push_str("): ");
        out.push_str(&format_value(attr.value()));
        out.push('\n');
    }

    for child in node.children() {
        write_md_node(child, out, child_indent);
    }
}

fn render_html(node: &Node, title: &str) -> String {
    let mut out = String::new();
    out.push_str("<!doctype html>\n");
    out.push_str("<html>\n<head>\n<meta charset=\"utf-8\">\n<title>");
    out.push_str(&escape_html(title));
    out.push_str(
        "</title>\n<style>\
body{font-family:\"JetBrains Mono\",\"Fira Code\",\"SFMono-Regular\",Menlo,Consolas,monospace;margin:16px;}\
ul{list-style:none;padding-left:16px;}\
.node-name{font-weight:600;}\
.attr-name{color:#0b5394;}\
.attr-type{color:#666;font-size:0.9em;margin-left:4px;}\
.attr-value{color:#222;}\
.attr{margin:2px 0;}\
</style>\n</head>\n<body>\n<ul class=\"cfg\">\n",
    );
    write_html_node(node, &mut out, 2);
    out.push_str("</ul>\n</body>\n</html>\n");
    out
}

fn write_html_node(node: &Node, out: &mut String, indent: usize) {
    push_indent(out, indent);
    out.push_str("<li class=\"node\"><span class=\"node-name\">");
    out.push_str(&escape_html(&format_name(node.name())));
    out.push_str("</span>");

    let has_attrs = !node.attrs().is_empty();
    let has_children = !node.children().is_empty();
    if has_attrs || has_children {
        out.push_str("\n");
        push_indent(out, indent + 2);
        out.push_str("<ul>\n");
        for attr in node.attrs() {
            push_indent(out, indent + 4);
            out.push_str("<li class=\"attr\"><span class=\"attr-name\">@");
            out.push_str(&escape_html(&format_name(attr.name())));
            out.push_str("</span><span class=\"attr-type\"> (");
            out.push_str(data_type_name(attr.value().data_type()));
            out.push_str(")</span> ");
            out.push_str("<span class=\"attr-value\">");
            out.push_str(&escape_html(&format_value(attr.value())));
            out.push_str("</span></li>\n");
        }
        for child in node.children() {
            write_html_node(child, out, indent + 4);
        }
        push_indent(out, indent + 2);
        out.push_str("</ul>\n");
        push_indent(out, indent);
    }
    out.push_str("</li>\n");
}

fn data_type_name(data_type: DataType) -> &'static str {
    match data_type {
        DataType::Unknown => "Unknown",
        DataType::Int => "Int",
        DataType::Real => "Real",
        DataType::Bool => "Bool",
        DataType::Str => "Str",
        DataType::Time => "Time",
        DataType::Sing => "Sing",
        DataType::Bin => "Bin",
        DataType::IntArray => "IntArray",
        DataType::RealArray => "RealArray",
        DataType::BoolArray => "BoolArray",
        DataType::StrArray => "StrArray",
        DataType::TimeArray => "TimeArray",
        DataType::SingArray => "SingArray",
        DataType::BinArray => "BinArray",
    }
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Unknown => "Unknown".to_string(),
        Value::Int(v) => v.to_string(),
        Value::Real(v) => format_float(*v),
        Value::Bool(v) => {
            if *v {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        Value::Str(data) => format_string(data),
        Value::Time(v) => format_time(*v),
        Value::Sing(v) => format_float(*v as f64),
        Value::Bin(data) => format_bin(data),
        Value::IntArray(values) => format_array(values, |v| v.to_string()),
        Value::RealArray(values) => format_array(values, |v| format_float(*v)),
        Value::BoolArray(values) => format_array(values, |v| {
            if *v == 0 {
                "False".to_string()
            } else {
                "True".to_string()
            }
        }),
        Value::StrArray(values) => format_array(values, |v| format_string(v)),
        Value::TimeArray(values) => format_array(values, |v| format_time(*v)),
        Value::SingArray(values) => format_array(values, |v| format_float(*v as f64)),
        Value::BinArray {
            count,
            record_size,
            data,
        } => format!(
            "bin_array(count={}, record_size={}, bytes={})",
            count,
            record_size,
            data.len()
        ),
    }
}

fn format_array<T, F>(values: &[T], to_string: F) -> String
where
    F: Fn(&T) -> String,
{
    let total = values.len();
    let shown = total.min(ARRAY_PREVIEW_LIMIT);
    let mut out = String::new();
    out.push('[');
    for (idx, value) in values.iter().take(shown).enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&to_string(value));
    }
    if total > shown {
        if shown > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("... +{}", total - shown));
    }
    out.push(']');
    out
}

fn format_string(data: &[u8]) -> String {
    format!("\"{}\"", escape_bytes(data))
}

fn format_bin(data: &[u8]) -> String {
    if data.is_empty() {
        return "bin[len=0]".to_string();
    }
    let shown = data.len().min(BIN_PREVIEW_BYTES);
    let mut sample = String::new();
    for (idx, byte) in data.iter().take(shown).enumerate() {
        if idx > 0 {
            sample.push(' ');
        }
        sample.push_str(&format!("{:02X}", byte));
    }
    if data.len() > shown {
        sample.push_str(" ...");
    }
    format!("bin[len={}, hex={}]", data.len(), sample)
}

fn format_time(value: f64) -> String {
    if !value.is_finite() {
        return "0".to_string();
    }
    let days = value.trunc();
    let frac = value - days;
    let millis = (frac * 86_400_000.0).round();
    let dt = delphi_epoch()
        + Duration::days(days as i64)
        + Duration::milliseconds(millis as i64);
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}

fn delphi_epoch() -> NaiveDateTime {
    NaiveDate::from_ymd_opt(1899, 12, 30)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

fn format_float(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            "-Inf".to_string()
        } else {
            "Inf".to_string()
        }
    } else {
        value.to_string()
    }
}

fn format_name(name: &[u8]) -> String {
    escape_bytes(name)
}

fn escape_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\\""),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7E => out.push(b as char),
            _ => out.push_str(&format!("\\x{:02X}", b)),
        }
    }
    out
}

fn escape_html(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push(' ');
    }
}
