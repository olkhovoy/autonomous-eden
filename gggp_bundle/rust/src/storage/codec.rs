use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use crate::storage::{Attr, DataType, Error, Node, Value};

const FILE_HEADER_V1: &[u8] =
    b"Custom file format. Contact author for specification. E-mail: <olkhovoy@gmail.com>";
const FILE_HEADER_V2: &[u8] =
    b"Custom file v.0002. Contact author for specification. E-mail: <olkhovoy@gmail.com>";
const NAME_MAX: u32 = 1024 * 1024;

type Result<T> = std::result::Result<T, Error>;

impl Node {
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Node> {
        read_tree(reader)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Node> {
        let mut cursor = Cursor::new(bytes);
        read_tree(&mut cursor)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Node> {
        let mut file = File::open(path)?;
        read_tree(&mut file)
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        write_tree(self, writer)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        write_tree(self, &mut buf)?;
        Ok(buf)
    }

    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = File::create(path)?;
        write_tree(self, &mut file)
    }
}

pub fn read_tree<R: Read>(reader: &mut R) -> Result<Node> {
    let mut header = vec![0u8; FILE_HEADER_V2.len()];
    reader.read_exact(&mut header)?;
    if header == FILE_HEADER_V2 {
        return read_tree_v2(reader);
    }
    if header == FILE_HEADER_V1 {
        return read_tree_v1(reader);
    }
    if header.starts_with(b"Custom file v.") {
        if let Some(version) = parse_header_version(&header) {
            if version >= 2 {
                return read_tree_v2(reader);
            }
        }
    }
    Err(Error::InvalidHeader)
}

pub fn write_tree<W: Write>(node: &Node, writer: &mut W) -> Result<()> {
    writer.write_all(FILE_HEADER_V2)?;
    let mut names = NameWriter::new();
    write_node_v2(node, writer, &mut names)
}

fn read_tree_v2<R: Read>(reader: &mut R) -> Result<Node> {
    let mut names = NameReader::new();
    let root_name = names
        .read_name(reader)?
        .ok_or(Error::InvalidFormat("missing root name"))?;
    read_node_v2(reader, &mut names, root_name)
}

fn read_tree_v1<R: Read>(reader: &mut R) -> Result<Node> {
    let names = read_names_v1(reader)?;
    read_node_v1(reader, &names, Vec::new())
}

fn parse_header_version(header: &[u8]) -> Option<u32> {
    if header.len() < 18 {
        return None;
    }
    let version_bytes = &header[14..18];
    let version_str = std::str::from_utf8(version_bytes).ok()?;
    version_str.parse::<u32>().ok()
}

fn read_names_v1<R: Read>(reader: &mut R) -> Result<Vec<Vec<u8>>> {
    let count = read_i32(reader)?;
    if count < 0 {
        return Err(Error::InvalidFormat("negative name count"));
    }
    let mut names = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let len = read_i32(reader)?;
        if len < 0 {
            return Err(Error::InvalidFormat("negative name length"));
        }
        let bytes = read_bytes(reader, len as u32)?;
        names.push(bytes);
    }
    Ok(names)
}

fn read_node_v1<R: Read>(reader: &mut R, names: &[Vec<u8>], name: Vec<u8>) -> Result<Node> {
    let attr_count = read_i32(reader)?;
    if attr_count < 0 {
        return Err(Error::InvalidFormat("negative attribute count"));
    }
    let data_len = read_i32(reader)?;
    if data_len < 0 {
        return Err(Error::InvalidFormat("negative data length"));
    }
    let mut data = vec![0u8; data_len as usize];
    reader.read_exact(&mut data)?;

    let mut attrs = Vec::with_capacity(attr_count as usize);
    let mut offset = 0usize;
    for _ in 0..attr_count {
        if offset + 3 > data.len() {
            return Err(Error::InvalidFormat("attribute data truncated"));
        }
        let name_index = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        let dtype = DataType::from_u8(data[offset + 2])
            .ok_or(Error::InvalidFormat("unknown data type"))?;
        offset += 3;
        let attr_name = names
            .get(name_index)
            .cloned()
            .ok_or(Error::InvalidFormat("name index out of range"))?;
        let value = read_value_from_slice_v1(&data, &mut offset, dtype)?;
        attrs.push(Attr::new(attr_name, value));
    }

    let child_count = read_i32(reader)?;
    if child_count < 0 {
        return Err(Error::InvalidFormat("negative child count"));
    }
    let mut sorted = true;
    let mut duplicates = 1;
    let mut children = Vec::new();
    if child_count > 0 {
        sorted = read_u8(reader)? != 0;
        duplicates = read_i32(reader)?;
        children = Vec::with_capacity(child_count as usize);
        for _ in 0..child_count {
            let name_idx = read_i32(reader)?;
            if name_idx < 0 {
                return Err(Error::InvalidFormat("negative node name index"));
            }
            let child_name = names
                .get(name_idx as usize)
                .cloned()
                .ok_or(Error::InvalidFormat("node name index out of range"))?;
            let child = read_node_v1(reader, names, child_name)?;
            children.push(child);
        }
    }

    Ok(Node::from_parts(
        name,
        0,
        0,
        sorted,
        duplicates,
        attrs,
        children,
    ))
}

fn read_node_v2<R: Read>(
    reader: &mut R,
    names: &mut NameReader,
    name: Vec<u8>,
) -> Result<Node> {
    let global_id = read_i32(reader)?;
    let mut attrs = Vec::new();
    loop {
        let attr_name = match names.read_name(reader)? {
            Some(name) => name,
            None => break,
        };
        let dtype_byte = read_u8(reader)?;
        let dtype = DataType::from_u8(dtype_byte)
            .ok_or(Error::InvalidFormat("unknown data type"))?;
        let value = read_value(reader, dtype)?;
        attrs.push(Attr::new(attr_name, value));
    }

    let sorted = read_u8(reader)? != 0;
    let duplicates = read_i32(reader)?;
    let modified = read_i32(reader)?;

    let mut children = Vec::new();
    loop {
        let child_name = match names.read_name(reader)? {
            Some(name) => name,
            None => break,
        };
        let child = read_node_v2(reader, names, child_name)?;
        children.push(child);
    }

    Ok(Node::from_parts(
        name,
        global_id,
        modified,
        sorted,
        duplicates,
        attrs,
        children,
    ))
}

fn write_node_v2<W: Write>(node: &Node, writer: &mut W, names: &mut NameWriter) -> Result<()> {
    names.write_name(writer, node.name())?;
    write_i32(writer, node.global_id)?;

    for attr in node.attrs() {
        names.write_name(writer, attr.name())?;
        write_value(writer, attr.value())?;
    }

    write_i32(writer, -1)?;

    write_u8(writer, if node.sorted() { 1 } else { 0 })?;
    write_i32(writer, node.duplicates())?;
    write_i32(writer, node.modified())?;

    for child in node.children() {
        write_node_v2(child, writer, names)?;
    }

    write_i32(writer, -1)?;
    Ok(())
}

fn read_value<R: Read>(reader: &mut R, dtype: DataType) -> Result<Value> {
    match dtype {
        DataType::Unknown => Ok(Value::Unknown),
        DataType::Int => Ok(Value::Int(read_i32(reader)?)),
        DataType::Real => Ok(Value::Real(read_f64(reader)?)),
        DataType::Bool => Ok(Value::Bool(read_u8(reader)? != 0)),
        DataType::Str => {
            let len = read_u32(reader)?;
            Ok(Value::Str(read_bytes(reader, len)?))
        }
        DataType::Time => Ok(Value::Time(read_f64(reader)?)),
        DataType::Sing => Ok(Value::Sing(read_f32(reader)?)),
        DataType::Bin => {
            let len = read_u32(reader)?;
            Ok(Value::Bin(read_bytes(reader, len)?))
        }
        DataType::IntArray => {
            let len = read_u32(reader)? as usize;
            Ok(Value::IntArray(read_i32_vec(reader, len)?))
        }
        DataType::RealArray => {
            let len = read_u32(reader)? as usize;
            Ok(Value::RealArray(read_f64_vec(reader, len)?))
        }
        DataType::BoolArray => {
            let len = read_u32(reader)?;
            Ok(Value::BoolArray(read_bytes(reader, len)?))
        }
        DataType::StrArray => {
            let count = read_u32(reader)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                let len = read_u32(reader)?;
                items.push(read_bytes(reader, len)?);
            }
            Ok(Value::StrArray(items))
        }
        DataType::TimeArray => {
            let len = read_u32(reader)? as usize;
            Ok(Value::TimeArray(read_f64_vec(reader, len)?))
        }
        DataType::SingArray => {
            let len = read_u32(reader)? as usize;
            Ok(Value::SingArray(read_f32_vec(reader, len)?))
        }
        DataType::BinArray => {
            let count = read_u32(reader)?;
            let record_size = read_u32(reader)?;
            let total = count
                .checked_mul(record_size)
                .ok_or(Error::LengthOverflow)?;
            let data = read_bytes(reader, total)?;
            Ok(Value::BinArray {
                count,
                record_size,
                data,
            })
        }
    }
}

fn read_value_from_slice_v1(
    data: &[u8],
    offset: &mut usize,
    dtype: DataType,
) -> Result<Value> {
    match dtype {
        DataType::Unknown => Ok(Value::Unknown),
        DataType::Int => Ok(Value::Int(read_i32_from_slice(data, offset)?)),
        DataType::Real => Ok(Value::Real(read_f64_from_slice(data, offset)?)),
        DataType::Bool => Ok(Value::Bool(read_u8_from_slice(data, offset)? != 0)),
        DataType::Str => {
            let len = read_u32_from_slice(data, offset)?;
            Ok(Value::Str(read_bytes_from_slice(data, offset, len)?))
        }
        DataType::Time => Ok(Value::Time(read_f64_from_slice(data, offset)?)),
        DataType::Sing => Ok(Value::Sing(read_f32_from_slice(data, offset)?)),
        DataType::Bin => {
            let len = read_u32_from_slice(data, offset)?;
            Ok(Value::Bin(read_bytes_from_slice(data, offset, len)?))
        }
        DataType::IntArray => {
            let len = read_u32_from_slice(data, offset)? as usize;
            Ok(Value::IntArray(read_i32_vec_from_slice(data, offset, len)?))
        }
        DataType::RealArray => {
            let len = read_u32_from_slice(data, offset)? as usize;
            Ok(Value::RealArray(read_f64_vec_from_slice(data, offset, len)?))
        }
        DataType::BoolArray => {
            let len = read_u32_from_slice(data, offset)?;
            Ok(Value::BoolArray(read_bytes_from_slice(data, offset, len)?))
        }
        DataType::StrArray => {
            let count = read_u32_from_slice(data, offset)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                let len = read_u32_from_slice(data, offset)?;
                items.push(read_bytes_from_slice(data, offset, len)?);
            }
            Ok(Value::StrArray(items))
        }
        DataType::TimeArray => {
            let len = read_u32_from_slice(data, offset)? as usize;
            Ok(Value::TimeArray(read_f64_vec_from_slice(data, offset, len)?))
        }
        DataType::SingArray => {
            let len = read_u32_from_slice(data, offset)? as usize;
            Ok(Value::SingArray(read_f32_vec_from_slice(data, offset, len)?))
        }
        DataType::BinArray => {
            let count = read_u32_from_slice(data, offset)?;
            let record_size = read_u32_from_slice(data, offset)?;
            let total = count
                .checked_mul(record_size)
                .ok_or(Error::LengthOverflow)?;
            let data = read_bytes_from_slice(data, offset, total)?;
            Ok(Value::BinArray {
                count,
                record_size,
                data,
            })
        }
    }
}

fn write_value<W: Write>(writer: &mut W, value: &Value) -> Result<()> {
    write_u8(writer, value.data_type() as u8)?;
    match value {
        Value::Unknown => Ok(()),
        Value::Int(v) => write_i32(writer, *v),
        Value::Real(v) => write_f64(writer, *v),
        Value::Bool(v) => write_u8(writer, if *v { 1 } else { 0 }),
        Value::Str(data) => {
            write_u32(writer, data.len() as u32)?;
            writer.write_all(data)?;
            Ok(())
        }
        Value::Time(v) => write_f64(writer, *v),
        Value::Sing(v) => write_f32(writer, *v),
        Value::Bin(data) => {
            write_u32(writer, data.len() as u32)?;
            writer.write_all(data)?;
            Ok(())
        }
        Value::IntArray(values) => {
            write_u32(writer, values.len() as u32)?;
            for v in values {
                write_i32(writer, *v)?;
            }
            Ok(())
        }
        Value::RealArray(values) => {
            write_u32(writer, values.len() as u32)?;
            for v in values {
                write_f64(writer, *v)?;
            }
            Ok(())
        }
        Value::BoolArray(values) => {
            write_u32(writer, values.len() as u32)?;
            writer.write_all(values)?;
            Ok(())
        }
        Value::StrArray(values) => {
            write_u32(writer, values.len() as u32)?;
            for item in values {
                write_u32(writer, item.len() as u32)?;
                writer.write_all(item)?;
            }
            Ok(())
        }
        Value::TimeArray(values) => {
            write_u32(writer, values.len() as u32)?;
            for v in values {
                write_f64(writer, *v)?;
            }
            Ok(())
        }
        Value::SingArray(values) => {
            write_u32(writer, values.len() as u32)?;
            for v in values {
                write_f32(writer, *v)?;
            }
            Ok(())
        }
        Value::BinArray {
            count,
            record_size,
            data,
        } => {
            let expected = (*count as usize)
                .checked_mul(*record_size as usize)
                .ok_or(Error::LengthOverflow)?;
            if data.len() != expected {
                return Err(Error::DataLengthMismatch {
                    expected,
                    actual: data.len(),
                });
            }
            write_u32(writer, *count)?;
            write_u32(writer, *record_size)?;
            writer.write_all(data)?;
            Ok(())
        }
    }
}

fn read_bytes<R: Read>(reader: &mut R, len: u32) -> Result<Vec<u8>> {
    let len = len as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i32_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<i32>> {
    let mut buf = vec![0u8; len * 4];
    reader.read_exact(&mut buf)?;
    let mut values = Vec::with_capacity(len);
    for chunk in buf.chunks_exact(4) {
        values.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn read_f64_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0u8; len * 8];
    reader.read_exact(&mut buf)?;
    let mut values = Vec::with_capacity(len);
    for chunk in buf.chunks_exact(8) {
        values.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    Ok(values)
}

fn read_f32_vec<R: Read>(reader: &mut R, len: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; len * 4];
    reader.read_exact(&mut buf)?;
    let mut values = Vec::with_capacity(len);
    for chunk in buf.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn write_i32<W: Write>(writer: &mut W, value: i32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_f32<W: Write>(writer: &mut W, value: f32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u8<W: Write>(writer: &mut W, value: u8) -> Result<()> {
    writer.write_all(&[value])?;
    Ok(())
}

fn read_i32_from_slice(data: &[u8], offset: &mut usize) -> Result<i32> {
    if *offset + 4 > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let value = i32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(value)
}

fn read_u32_from_slice(data: &[u8], offset: &mut usize) -> Result<u32> {
    if *offset + 4 > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let value = u32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(value)
}

fn read_f64_from_slice(data: &[u8], offset: &mut usize) -> Result<f64> {
    if *offset + 8 > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let value = f64::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
        data[*offset + 4],
        data[*offset + 5],
        data[*offset + 6],
        data[*offset + 7],
    ]);
    *offset += 8;
    Ok(value)
}

fn read_f32_from_slice(data: &[u8], offset: &mut usize) -> Result<f32> {
    if *offset + 4 > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let value = f32::from_le_bytes([
        data[*offset],
        data[*offset + 1],
        data[*offset + 2],
        data[*offset + 3],
    ]);
    *offset += 4;
    Ok(value)
}

fn read_u8_from_slice(data: &[u8], offset: &mut usize) -> Result<u8> {
    if *offset + 1 > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let value = data[*offset];
    *offset += 1;
    Ok(value)
}

fn read_bytes_from_slice(data: &[u8], offset: &mut usize, len: u32) -> Result<Vec<u8>> {
    let len = len as usize;
    if *offset + len > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let out = data[*offset..*offset + len].to_vec();
    *offset += len;
    Ok(out)
}

fn read_i32_vec_from_slice(data: &[u8], offset: &mut usize, len: usize) -> Result<Vec<i32>> {
    let byte_len = len.checked_mul(4).ok_or(Error::LengthOverflow)?;
    if *offset + byte_len > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let mut values = Vec::with_capacity(len);
    for chunk in data[*offset..*offset + byte_len].chunks_exact(4) {
        values.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *offset += byte_len;
    Ok(values)
}

fn read_f64_vec_from_slice(data: &[u8], offset: &mut usize, len: usize) -> Result<Vec<f64>> {
    let byte_len = len.checked_mul(8).ok_or(Error::LengthOverflow)?;
    if *offset + byte_len > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let mut values = Vec::with_capacity(len);
    for chunk in data[*offset..*offset + byte_len].chunks_exact(8) {
        values.push(f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]));
    }
    *offset += byte_len;
    Ok(values)
}

fn read_f32_vec_from_slice(data: &[u8], offset: &mut usize, len: usize) -> Result<Vec<f32>> {
    let byte_len = len.checked_mul(4).ok_or(Error::LengthOverflow)?;
    if *offset + byte_len > data.len() {
        return Err(Error::InvalidFormat("buffer too small"));
    }
    let mut values = Vec::with_capacity(len);
    for chunk in data[*offset..*offset + byte_len].chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *offset += byte_len;
    Ok(values)
}

struct NameReader {
    names: Vec<Vec<u8>>,
}

impl NameReader {
    fn new() -> Self {
        NameReader { names: Vec::new() }
    }

    fn read_name<R: Read>(&mut self, reader: &mut R) -> Result<Option<Vec<u8>>> {
        let id = read_i32(reader)?;
        if id == -1 {
            return Ok(None);
        }
        if id <= 0 {
            return Err(Error::InvalidNameId(id));
        }
        let idx = (id - 1) as usize;
        if idx < self.names.len() {
            return Ok(Some(self.names[idx].clone()));
        }
        if idx != self.names.len() {
            return Err(Error::InvalidNameId(id));
        }
        let len = read_u32(reader)?;
        if len > NAME_MAX {
            return Err(Error::NameTooLarge(len));
        }
        let name = read_bytes(reader, len)?;
        self.names.push(name.clone());
        Ok(Some(name))
    }
}

struct NameWriter {
    map: std::collections::HashMap<Vec<u8>, i32>,
    next_id: i32,
}

impl NameWriter {
    fn new() -> Self {
        NameWriter {
            map: std::collections::HashMap::new(),
            next_id: 1,
        }
    }

    fn write_name<W: Write>(&mut self, writer: &mut W, name: &[u8]) -> Result<()> {
        let key = crate::storage::normalize_name(name);
        if let Some(id) = self.map.get(&key).copied() {
            write_i32(writer, id)?;
            return Ok(());
        }
        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(key, id);
        write_i32(writer, id)?;
        write_u32(writer, name.len() as u32)?;
        writer.write_all(name)?;
        Ok(())
    }
}
