use chrono::{Duration, NaiveDate, NaiveDateTime};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Unknown = 0,
    Int = 1,
    Real = 2,
    Bool = 3,
    Str = 4,
    Time = 5,
    Sing = 6,
    Bin = 7,
    IntArray = 8,
    RealArray = 9,
    BoolArray = 10,
    StrArray = 11,
    TimeArray = 12,
    SingArray = 13,
    BinArray = 14,
}

impl DataType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(DataType::Unknown),
            1 => Some(DataType::Int),
            2 => Some(DataType::Real),
            3 => Some(DataType::Bool),
            4 => Some(DataType::Str),
            5 => Some(DataType::Time),
            6 => Some(DataType::Sing),
            7 => Some(DataType::Bin),
            8 => Some(DataType::IntArray),
            9 => Some(DataType::RealArray),
            10 => Some(DataType::BoolArray),
            11 => Some(DataType::StrArray),
            12 => Some(DataType::TimeArray),
            13 => Some(DataType::SingArray),
            14 => Some(DataType::BinArray),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unknown,
    Int(i32),
    Real(f64),
    Bool(bool),
    Str(Vec<u8>),
    Time(f64),
    Sing(f32),
    Bin(Vec<u8>),
    IntArray(Vec<i32>),
    RealArray(Vec<f64>),
    BoolArray(Vec<u8>),
    StrArray(Vec<Vec<u8>>),
    TimeArray(Vec<f64>),
    SingArray(Vec<f32>),
    BinArray { count: u32, record_size: u32, data: Vec<u8> },
}

impl Value {
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Unknown => DataType::Unknown,
            Value::Int(_) => DataType::Int,
            Value::Real(_) => DataType::Real,
            Value::Bool(_) => DataType::Bool,
            Value::Str(_) => DataType::Str,
            Value::Time(_) => DataType::Time,
            Value::Sing(_) => DataType::Sing,
            Value::Bin(_) => DataType::Bin,
            Value::IntArray(_) => DataType::IntArray,
            Value::RealArray(_) => DataType::RealArray,
            Value::BoolArray(_) => DataType::BoolArray,
            Value::StrArray(_) => DataType::StrArray,
            Value::TimeArray(_) => DataType::TimeArray,
            Value::SingArray(_) => DataType::SingArray,
            Value::BinArray { .. } => DataType::BinArray,
        }
    }

    pub fn as_int(&self) -> i32 {
        match self {
            Value::Int(value) => *value,
            Value::Bool(value) => if *value { 1 } else { 0 },
            Value::Sing(value) => value.round() as i32,
            Value::Str(value) => parse_int_bytes(value).unwrap_or(0),
            Value::Real(value) | Value::Time(value) => value.round() as i32,
            Value::Bin(value) => value.len() as i32,
            Value::IntArray(value) => value.len() as i32,
            Value::RealArray(value) => value.len() as i32,
            Value::BoolArray(value) => value.len() as i32,
            Value::StrArray(value) => value.len() as i32,
            Value::TimeArray(value) => value.len() as i32,
            Value::SingArray(value) => value.len() as i32,
            Value::BinArray { count, .. } => *count as i32,
            Value::Unknown => 0,
        }
    }

    pub fn as_real(&self) -> f64 {
        match self {
            Value::Int(value) => *value as f64,
            Value::Bool(value) => if *value { 1.0 } else { 0.0 },
            Value::Sing(value) => *value as f64,
            Value::Str(value) => parse_float_bytes(value).unwrap_or(0.0),
            Value::Real(value) | Value::Time(value) => *value,
            Value::Bin(value) => value.len() as f64,
            Value::IntArray(value) => value.len() as f64,
            Value::RealArray(value) => value.len() as f64,
            Value::BoolArray(value) => value.len() as f64,
            Value::StrArray(value) => value.len() as f64,
            Value::TimeArray(value) => value.len() as f64,
            Value::SingArray(value) => value.len() as f64,
            Value::BinArray { count, .. } => *count as f64,
            Value::Unknown => 0.0,
        }
    }

    pub fn as_sing(&self) -> f32 {
        match self {
            Value::Int(value) => *value as f32,
            Value::Bool(value) => if *value { 1.0 } else { 0.0 },
            Value::Sing(value) => *value,
            Value::Str(value) => parse_float_bytes(value).unwrap_or(0.0) as f32,
            Value::Real(value) | Value::Time(value) => *value as f32,
            Value::Bin(value) => value.len() as f32,
            Value::IntArray(value) => value.len() as f32,
            Value::RealArray(value) => value.len() as f32,
            Value::BoolArray(value) => value.len() as f32,
            Value::StrArray(value) => value.len() as f32,
            Value::TimeArray(value) => value.len() as f32,
            Value::SingArray(value) => value.len() as f32,
            Value::BinArray { count, .. } => *count as f32,
            Value::Unknown => 0.0,
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Int(value) => *value > 0,
            Value::Bool(value) => *value,
            Value::Sing(value) => *value != 0.0,
            Value::Real(value) | Value::Time(value) => *value > 0.0,
            Value::Str(value) => !value.is_empty(),
            Value::Bin(value) => !value.is_empty(),
            Value::IntArray(value) => !value.is_empty(),
            Value::RealArray(value) => !value.is_empty(),
            Value::BoolArray(value) => !value.is_empty(),
            Value::StrArray(value) => !value.is_empty(),
            Value::TimeArray(value) => !value.is_empty(),
            Value::SingArray(value) => !value.is_empty(),
            Value::BinArray { count, .. } => *count > 0,
            Value::Unknown => false,
        }
    }

    pub fn as_time(&self) -> f64 {
        match self {
            Value::Int(value) => *value as f64,
            Value::Bool(value) => if *value { 1.0 } else { 0.0 },
            Value::Sing(value) => *value as f64,
            Value::Str(value) => delphi_string_to_datetime(value).unwrap_or(0.0),
            Value::Real(value) | Value::Time(value) => *value,
            Value::Bin(value) => value.len() as f64,
            Value::IntArray(value) => value.len() as f64,
            Value::RealArray(value) => value.len() as f64,
            Value::BoolArray(value) => value.len() as f64,
            Value::StrArray(value) => value.len() as f64,
            Value::TimeArray(value) => value.len() as f64,
            Value::SingArray(value) => value.len() as f64,
            Value::BinArray { count, .. } => *count as f64,
            Value::Unknown => 0.0,
        }
    }

    pub fn as_str_bytes(&self) -> Vec<u8> {
        match self {
            Value::Int(value) => value.to_string().into_bytes(),
            Value::Bool(value) => {
                if *value {
                    b"True".to_vec()
                } else {
                    b"False".to_vec()
                }
            }
            Value::Sing(value) => value.to_string().into_bytes(),
            Value::Str(value) => value.clone(),
            Value::Real(value) => value.to_string().into_bytes(),
            Value::Time(value) => delphi_datetime_to_string(*value),
            Value::Bin(value) => value.len().to_string().into_bytes(),
            Value::IntArray(value) => value.len().to_string().into_bytes(),
            Value::RealArray(value) => value.len().to_string().into_bytes(),
            Value::BoolArray(value) => value.len().to_string().into_bytes(),
            Value::StrArray(value) => value.len().to_string().into_bytes(),
            Value::TimeArray(value) => value.len().to_string().into_bytes(),
            Value::SingArray(value) => value.len().to_string().into_bytes(),
            Value::BinArray { count, .. } => count.to_string().into_bytes(),
            Value::Unknown => Vec::new(),
        }
    }

    pub fn as_int_array(&self) -> Vec<i32> {
        match self {
            Value::Int(value) => vec![*value],
            Value::Bool(value) => vec![if *value { 1 } else { 0 }],
            Value::Sing(value) => vec![value.round() as i32],
            Value::Str(value) => vec![parse_int_bytes(value).unwrap_or(0)],
            Value::Real(value) | Value::Time(value) => vec![value.round() as i32],
            Value::Bin(value) => vec![value.len() as i32],
            Value::IntArray(value) => value.clone(),
            Value::RealArray(value) => value.iter().map(|v| v.round() as i32).collect(),
            Value::BoolArray(value) => value.iter().map(|v| if *v == 0 { 0 } else { 1 }).collect(),
            Value::StrArray(value) => value
                .iter()
                .map(|v| parse_int_bytes(v).unwrap_or(0))
                .collect(),
            Value::TimeArray(value) => value.iter().map(|v| v.round() as i32).collect(),
            Value::SingArray(value) => value.iter().map(|v| v.round() as i32).collect(),
            Value::BinArray { count, record_size, .. } => {
                vec![*record_size as i32; *count as usize]
            }
            Value::Unknown => Vec::new(),
        }
    }

    pub fn as_real_array(&self) -> Vec<f64> {
        match self {
            Value::Int(value) => vec![*value as f64],
            Value::Bool(value) => vec![if *value { 1.0 } else { 0.0 }],
            Value::Sing(value) => vec![*value as f64],
            Value::Str(value) => vec![parse_float_bytes(value).unwrap_or(0.0)],
            Value::Real(value) | Value::Time(value) => vec![*value],
            Value::Bin(value) => vec![value.len() as f64],
            Value::IntArray(value) => value.iter().map(|v| *v as f64).collect(),
            Value::RealArray(value) => value.clone(),
            Value::BoolArray(value) => value.iter().map(|v| if *v == 0 { 0.0 } else { 1.0 }).collect(),
            Value::StrArray(value) => value
                .iter()
                .map(|v| parse_float_bytes(v).unwrap_or(0.0))
                .collect(),
            Value::TimeArray(value) => value.clone(),
            Value::SingArray(value) => value.iter().map(|v| *v as f64).collect(),
            Value::BinArray { count, record_size, .. } => {
                vec![*record_size as f64; *count as usize]
            }
            Value::Unknown => Vec::new(),
        }
    }

    pub fn as_str_array(&self) -> Vec<Vec<u8>> {
        match self {
            Value::Int(value) => vec![value.to_string().into_bytes()],
            Value::Bool(value) => vec![if *value { b"True".to_vec() } else { b"False".to_vec() }],
            Value::Sing(value) => vec![value.to_string().into_bytes()],
            Value::Str(value) => vec![value.clone()],
            Value::Real(value) => vec![value.to_string().into_bytes()],
            Value::Time(value) => vec![delphi_datetime_to_string(*value)],
            Value::Bin(value) => vec![value.len().to_string().into_bytes()],
            Value::IntArray(value) => value.iter().map(|v| v.to_string().into_bytes()).collect(),
            Value::RealArray(value) => value.iter().map(|v| v.to_string().into_bytes()).collect(),
            Value::BoolArray(value) => value
                .iter()
                .map(|v| if *v == 0 { b"False".to_vec() } else { b"True".to_vec() })
                .collect(),
            Value::StrArray(value) => value.clone(),
            Value::TimeArray(value) => value.iter().map(|v| delphi_datetime_to_string(*v)).collect(),
            Value::SingArray(value) => value.iter().map(|v| v.to_string().into_bytes()).collect(),
            Value::BinArray { count, record_size, .. } => {
                vec![record_size.to_string().into_bytes(); *count as usize]
            }
            Value::Unknown => Vec::new(),
        }
    }

    pub fn as_time_array(&self) -> Vec<f64> {
        match self {
            Value::TimeArray(value) => value.clone(),
            Value::RealArray(value) => value.clone(),
            Value::IntArray(value) => value.iter().map(|v| *v as f64).collect(),
            Value::StrArray(value) => value
                .iter()
                .map(|v| delphi_string_to_datetime(v).unwrap_or(0.0))
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_bool_array(&self) -> Vec<bool> {
        match self {
            Value::BoolArray(value) => value.iter().map(|v| *v != 0).collect(),
            Value::IntArray(value) => value.iter().map(|v| *v != 0).collect(),
            _ => Vec::new(),
        }
    }

    pub fn as_sing_array(&self) -> Vec<f32> {
        match self {
            Value::SingArray(value) => value.clone(),
            Value::RealArray(value) => value.iter().map(|v| *v as f32).collect(),
            Value::IntArray(value) => value.iter().map(|v| *v as f32).collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    name: Vec<u8>,
    value: Value,
}

impl Attr {
    pub fn new(name: impl Into<Vec<u8>>, value: Value) -> Self {
        Attr {
            name: name.into(),
            value,
        }
    }

    pub fn name(&self) -> &[u8] {
        &self.name
    }

    pub fn value(&self) -> &Value {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }
}

fn parse_int_bytes(bytes: &[u8]) -> Option<i32> {
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    if idx >= bytes.len() {
        return None;
    }
    let mut sign: i64 = 1;
    if bytes[idx] == b'+' {
        idx += 1;
    } else if bytes[idx] == b'-' {
        sign = -1;
        idx += 1;
    }
    let mut value: i64 = 0;
    let mut has_digit = false;
    while idx < bytes.len() {
        let b = bytes[idx];
        if !b.is_ascii_digit() {
            break;
        }
        value = value * 10 + (b - b'0') as i64;
        has_digit = true;
        idx += 1;
    }
    if !has_digit {
        return None;
    }
    let signed = value * sign;
    if signed > i32::MAX as i64 || signed < i32::MIN as i64 {
        return None;
    }
    Some(signed as i32)
}

fn parse_float_bytes(bytes: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(bytes).ok()?.trim();
    if s.is_empty() {
        return None;
    }
    s.parse::<f64>().ok()
}

fn delphi_epoch() -> NaiveDateTime {
    NaiveDate::from_ymd_opt(1899, 12, 30)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
}

fn delphi_datetime_to_string(value: f64) -> Vec<u8> {
    if !value.is_finite() {
        return b"0".to_vec();
    }
    let days = value.trunc();
    let frac = value - days;
    let millis = (frac * 86_400_000.0).round();
    let dt = delphi_epoch()
        + Duration::days(days as i64)
        + Duration::milliseconds(millis as i64);
    dt.format("%Y-%m-%d %H:%M:%S").to_string().into_bytes()
}

fn delphi_string_to_datetime(bytes: &[u8]) -> Option<f64> {
    let s = std::str::from_utf8(bytes).ok()?.trim();
    if s.is_empty() {
        return None;
    }
    let dt = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").ok()?;
    let duration = dt.signed_duration_since(delphi_epoch());
    Some(duration.num_milliseconds() as f64 / 86_400_000.0)
}
