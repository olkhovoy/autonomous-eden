use std::collections::HashMap;

use crate::storage::{normalize_name, Attr, DataType, Error, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    name: Vec<u8>,
    pub(crate) global_id: i32,
    pub(crate) modified: i32,
    pub(crate) sorted: bool,
    pub(crate) duplicates: i32,
    pub(crate) attrs: Vec<Attr>,
    pub(crate) children: Vec<Node>,
    attr_index: HashMap<Vec<u8>, usize>,
    child_index: HashMap<Vec<u8>, usize>,
}

impl Node {
    pub fn new(name: impl Into<Vec<u8>>) -> Self {
        Node {
            name: name.into(),
            global_id: 0,
            modified: 0,
            sorted: true,
            duplicates: 1,
            attrs: Vec::new(),
            children: Vec::new(),
            attr_index: HashMap::new(),
            child_index: HashMap::new(),
        }
    }

    pub fn name(&self) -> &[u8] {
        &self.name
    }

    pub fn set_name(&mut self, name: impl Into<Vec<u8>>) {
        self.name = name.into();
    }

    pub fn global_id(&self) -> i32 {
        self.global_id
    }

    pub fn set_global_id(&mut self, value: i32) {
        self.global_id = value;
    }

    pub fn modified(&self) -> i32 {
        self.modified
    }

    pub fn set_modified(&mut self, value: i32) {
        self.modified = value;
    }

    pub fn sorted(&self) -> bool {
        self.sorted
    }

    pub fn set_sorted(&mut self, value: bool) {
        self.sorted = value;
    }

    pub fn duplicates(&self) -> i32 {
        self.duplicates
    }

    pub fn set_duplicates(&mut self, value: i32) {
        self.duplicates = value;
    }

    pub fn attrs(&self) -> &[Attr] {
        &self.attrs
    }

    pub fn children(&self) -> &[Node] {
        &self.children
    }

    pub fn children_mut(&mut self) -> &mut Vec<Node> {
        &mut self.children
    }

    pub fn attr_count(&self) -> usize {
        self.attrs.len()
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    pub fn attr_name(&self, index: usize) -> Option<&[u8]> {
        self.attrs.get(index).map(|attr| attr.name())
    }

    pub fn attr_exists(&self, name: impl AsRef<[u8]>) -> bool {
        let key = normalize_name(name.as_ref());
        self.attr_index.contains_key(&key)
    }

    pub fn attr_type(&self, name: impl AsRef<[u8]>) -> Option<DataType> {
        self.get_value(name).map(|value| value.data_type())
    }

    pub fn get_value(&self, name: impl AsRef<[u8]>) -> Option<&Value> {
        let key = normalize_name(name.as_ref());
        self.attr_index
            .get(&key)
            .and_then(|idx| self.attrs.get(*idx))
            .map(|attr| attr.value())
    }

    pub fn get_value_mut(&mut self, name: impl AsRef<[u8]>) -> Option<&mut Value> {
        let key = normalize_name(name.as_ref());
        if let Some(idx) = self.attr_index.get(&key).copied() {
            return self.attrs.get_mut(idx).map(|attr| attr.value_mut());
        }
        None
    }

    pub fn set_value(&mut self, name: impl Into<Vec<u8>>, value: Value) {
        let name_vec = name.into();
        let key = normalize_name(&name_vec);
        if let Some(idx) = self.attr_index.get(&key).copied() {
            if self.attrs[idx].value() != &value {
                *self.attrs[idx].value_mut() = value;
                self.modified = self.modified.wrapping_add(1);
            }
            return;
        }
        let idx = self.attrs.len();
        self.attrs.push(Attr::new(name_vec, value));
        self.attr_index.insert(key, idx);
        self.modified = self.modified.wrapping_add(1);
    }

    pub fn remove_attr(&mut self, name: impl AsRef<[u8]>) -> bool {
        let key = normalize_name(name.as_ref());
        let idx = match self.attr_index.get(&key).copied() {
            Some(idx) => idx,
            None => return false,
        };
        self.attrs.remove(idx);
        self.rebuild_attr_index();
        self.modified = self.modified.wrapping_add(1);
        true
    }

    pub fn get_int(&self, name: impl AsRef<[u8]>) -> i32 {
        self.get_value(name).map_or(0, |value| value.as_int())
    }

    pub fn get_real(&self, name: impl AsRef<[u8]>) -> f64 {
        self.get_value(name).map_or(0.0, |value| value.as_real())
    }

    pub fn get_sing(&self, name: impl AsRef<[u8]>) -> f32 {
        self.get_value(name).map_or(0.0, |value| value.as_sing())
    }

    pub fn get_bool(&self, name: impl AsRef<[u8]>) -> bool {
        self.get_value(name).map_or(false, |value| value.as_bool())
    }

    pub fn get_time(&self, name: impl AsRef<[u8]>) -> f64 {
        self.get_value(name).map_or(0.0, |value| value.as_time())
    }

    pub fn get_str(&self, name: impl AsRef<[u8]>) -> Vec<u8> {
        self.get_value(name)
            .map(|value| value.as_str_bytes())
            .unwrap_or_default()
    }

    pub fn get_int_array(&self, name: impl AsRef<[u8]>) -> Vec<i32> {
        self.get_value(name)
            .map(|value| value.as_int_array())
            .unwrap_or_default()
    }

    pub fn get_real_array(&self, name: impl AsRef<[u8]>) -> Vec<f64> {
        self.get_value(name)
            .map(|value| value.as_real_array())
            .unwrap_or_default()
    }

    pub fn get_str_array(&self, name: impl AsRef<[u8]>) -> Vec<Vec<u8>> {
        self.get_value(name)
            .map(|value| value.as_str_array())
            .unwrap_or_default()
    }

    pub fn get_bin(&self, name: impl AsRef<[u8]>) -> Option<&[u8]> {
        match self.get_value(name) {
            Some(Value::Bin(data)) => Some(data),
            _ => None,
        }
    }

    pub fn get_bin_array(&self, name: impl AsRef<[u8]>) -> Option<(u32, u32, &[u8])> {
        match self.get_value(name) {
            Some(Value::BinArray {
                count,
                record_size,
                data,
            }) => Some((*count, *record_size, data)),
            _ => None,
        }
    }

    pub fn set_int(&mut self, name: impl Into<Vec<u8>>, value: i32) {
        self.set_value(name, Value::Int(value));
    }

    pub fn set_real(&mut self, name: impl Into<Vec<u8>>, value: f64) {
        self.set_value(name, Value::Real(value));
    }

    pub fn set_sing(&mut self, name: impl Into<Vec<u8>>, value: f32) {
        self.set_value(name, Value::Sing(value));
    }

    pub fn set_bool(&mut self, name: impl Into<Vec<u8>>, value: bool) {
        self.set_value(name, Value::Bool(value));
    }

    pub fn set_str(&mut self, name: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) {
        self.set_value(name, Value::Str(value.into()));
    }

    pub fn set_time(&mut self, name: impl Into<Vec<u8>>, value: f64) {
        self.set_value(name, Value::Time(value));
    }

    pub fn set_bin(&mut self, name: impl Into<Vec<u8>>, value: Vec<u8>) {
        self.set_value(name, Value::Bin(value));
    }

    pub fn set_int_array(&mut self, name: impl Into<Vec<u8>>, value: Vec<i32>) {
        self.set_value(name, Value::IntArray(value));
    }

    pub fn set_real_array(&mut self, name: impl Into<Vec<u8>>, value: Vec<f64>) {
        self.set_value(name, Value::RealArray(value));
    }

    pub fn set_bool_array(&mut self, name: impl Into<Vec<u8>>, value: Vec<bool>) {
        let data = value.into_iter().map(|v| if v { 1 } else { 0 }).collect();
        self.set_value(name, Value::BoolArray(data));
    }

    pub fn set_str_array(&mut self, name: impl Into<Vec<u8>>, value: Vec<Vec<u8>>) {
        self.set_value(name, Value::StrArray(value));
    }

    pub fn set_time_array(&mut self, name: impl Into<Vec<u8>>, value: Vec<f64>) {
        self.set_value(name, Value::TimeArray(value));
    }

    pub fn set_sing_array(&mut self, name: impl Into<Vec<u8>>, value: Vec<f32>) {
        self.set_value(name, Value::SingArray(value));
    }

    pub fn set_bin_array(
        &mut self,
        name: impl Into<Vec<u8>>,
        count: u32,
        record_size: u32,
        data: Vec<u8>,
    ) -> Result<(), Error> {
        let expected = count
            .checked_mul(record_size)
            .ok_or(Error::LengthOverflow)? as usize;
        if data.len() != expected {
            return Err(Error::DataLengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        self.set_value(
            name,
            Value::BinArray {
                count,
                record_size,
                data,
            },
        );
        Ok(())
    }

    pub fn child(&self, name: impl AsRef<[u8]>) -> Option<&Node> {
        let key = normalize_name(name.as_ref());
        self.child_index
            .get(&key)
            .and_then(|idx| self.children.get(*idx))
    }

    pub fn child_mut(&mut self, name: impl AsRef<[u8]>) -> Option<&mut Node> {
        let key = normalize_name(name.as_ref());
        if let Some(idx) = self.child_index.get(&key).copied() {
            return self.children.get_mut(idx);
        }
        None
    }

    pub fn child_by_index(&self, index: usize) -> Option<&Node> {
        self.children.get(index)
    }

    pub fn child_by_index_mut(&mut self, index: usize) -> Option<&mut Node> {
        self.children.get_mut(index)
    }

    pub fn get_or_create_child(&mut self, name: impl Into<Vec<u8>>) -> &mut Node {
        let name_vec = name.into();
        let key = normalize_name(&name_vec);
        if let Some(idx) = self.child_index.get(&key).copied() {
            return &mut self.children[idx];
        }
        let idx = self.children.len();
        self.children.push(Node::new(name_vec));
        self.child_index.insert(key, idx);
        self.modified = self.modified.wrapping_add(1);
        &mut self.children[idx]
    }

    pub fn add_child(&mut self, child: Node) -> &mut Node {
        let key = normalize_name(child.name());
        let idx = self.children.len();
        self.children.push(child);
        self.child_index.entry(key).or_insert(idx);
        self.modified = self.modified.wrapping_add(1);
        &mut self.children[idx]
    }

    pub fn clear_children(&mut self) {
        if !self.children.is_empty() {
            self.children.clear();
            self.child_index.clear();
            self.modified = self.modified.wrapping_add(1);
        }
    }

    pub fn clear_attrs(&mut self) {
        if !self.attrs.is_empty() {
            self.attrs.clear();
            self.attr_index.clear();
            self.modified = self.modified.wrapping_add(1);
        }
    }

    pub fn clear_all(&mut self) {
        self.clear_attrs();
        self.clear_children();
    }

    pub fn copy_from(&mut self, other: &Node, copy_name: bool, copy_children: bool) {
        if copy_name {
            self.name = other.name.clone();
        }
        self.attrs = other.attrs.clone();
        self.attr_index = other.attr_index.clone();
        if copy_children {
            self.children = other.children.clone();
            self.child_index = other.child_index.clone();
        } else {
            self.children.clear();
            self.child_index.clear();
        }
        self.sorted = other.sorted;
        self.duplicates = other.duplicates;
        self.modified = self.modified.wrapping_add(1);
    }

    pub(crate) fn from_parts(
        name: Vec<u8>,
        global_id: i32,
        modified: i32,
        sorted: bool,
        duplicates: i32,
        attrs: Vec<Attr>,
        children: Vec<Node>,
    ) -> Self {
        let mut node = Node {
            name,
            global_id,
            modified,
            sorted,
            duplicates,
            attrs,
            children,
            attr_index: HashMap::new(),
            child_index: HashMap::new(),
        };
        node.rebuild_indexes();
        node
    }

    pub(crate) fn rebuild_indexes(&mut self) {
        self.rebuild_attr_index();
        self.rebuild_child_index();
    }

    fn rebuild_attr_index(&mut self) {
        self.attr_index.clear();
        for (idx, attr) in self.attrs.iter().enumerate() {
            let key = normalize_name(attr.name());
            self.attr_index.entry(key).or_insert(idx);
        }
    }

    fn rebuild_child_index(&mut self) {
        self.child_index.clear();
        for (idx, child) in self.children.iter().enumerate() {
            let key = normalize_name(child.name());
            self.child_index.entry(key).or_insert(idx);
        }
    }
}
