use rand::Rng;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::io::{self, Read, Write};
use std::rc::{Rc, Weak};

pub mod phenotype;
use phenotype::{Phenotype, OutputType, VectorPhenotype, VectorSymbol};

pub mod vector;
use vector::compile_tree_to_vector;

pub mod hybrid;

use crate::storage::Node;

#[derive(Debug)]
pub enum GggpError {
    InvalidConfig(&'static str),
    InvalidData(String),
    Io(io::Error),
    MissingStart,
    MissingRules,
    MissingSymbol(String),
    InvalidChoice(String),
}

impl From<io::Error> for GggpError {
    fn from(err: io::Error) -> Self {
        GggpError::Io(err)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizeDirection {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone)]
pub struct FitnessHistoryRec {
    pub success: bool,
    pub fitness: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeType {
    Empty,
    Choice,
}

#[derive(Debug)]
pub struct GpRef {
    choice: Weak<GpChoice>,
    number: i32,
    symbol_name: String,
    symbol: RefCell<Option<Weak<GpSymbol>>>,
    pos: i32,
    len: i32,
    param_max_depth: i32,
    param_name: String,
    param_from: f64,
    param_to: f64,
    param_inc: f64,
    param_optimize: bool,
}

impl GpRef {
    pub fn number(&self) -> i32 {
        self.number
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub fn symbol(&self) -> Rc<GpSymbol> {
        self.symbol
            .borrow()
            .as_ref()
            .and_then(|sym| sym.upgrade())
            .expect("reference symbol is not assigned")
    }

    pub fn choice(&self) -> Rc<GpChoice> {
        self.choice
            .upgrade()
            .expect("reference choice is not assigned")
    }

    pub fn pos(&self) -> i32 {
        self.pos
    }

    pub fn len(&self) -> i32 {
        self.len
    }

    pub fn param_max_depth(&self) -> i32 {
        self.param_max_depth
    }

    pub fn param_inc(&self) -> f64 {
        self.param_inc
    }

    fn set_symbol(&self, symbol: &Rc<GpSymbol>) {
        *self.symbol.borrow_mut() = Some(Rc::downgrade(symbol));
    }
}

#[derive(Debug)]
pub struct GpChoice {
    symbol: Weak<GpSymbol>,
    number: i32,
    text: String,
    min_depth: i32,
    refs: Vec<Rc<GpRef>>,
    ref_map: HashMap<i32, Rc<GpRef>>,
}

impl GpChoice {
    pub fn number(&self) -> i32 {
        self.number
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn min_depth(&self) -> i32 {
        self.min_depth
    }

    pub fn symbol(&self) -> Rc<GpSymbol> {
        self.symbol
            .upgrade()
            .expect("choice symbol is not assigned")
    }

    pub fn refs(&self) -> &[Rc<GpRef>] {
        &self.refs
    }

    pub fn ref_by_number(&self, number: i32) -> Option<Rc<GpRef>> {
        self.ref_map.get(&number).cloned()
    }

    pub fn index_of_ref_number(&self, number: i32) -> Option<usize> {
        self.refs.iter().position(|r| r.number() == number)
    }

    pub fn matches(&self, symbols: &[Rc<GpSymbol>]) -> bool {
        if symbols.len() != self.refs.len() {
            return false;
        }
        for (idx, sym) in symbols.iter().enumerate() {
            let ref_sym = self.refs[idx].symbol();
            if !Rc::ptr_eq(&ref_sym, sym) {
                return false;
            }
        }
        true
    }
}

#[derive(Debug)]
pub struct GpSymbol {
    name: String,
    min_depth: i32,
    terminal: bool,
    choices: Vec<Rc<GpChoice>>,
    choice_map: HashMap<i32, Rc<GpChoice>>,
    max_depth: i32,
    max_crossover_nodes: i32,
    max_mutation_nodes: i32,
}

impl GpSymbol {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn min_depth(&self) -> i32 {
        self.min_depth
    }

    pub fn terminal(&self) -> bool {
        self.terminal
    }

    pub fn choices(&self) -> &[Rc<GpChoice>] {
        &self.choices
    }

    pub fn choice_by_number(&self, number: i32) -> Option<Rc<GpChoice>> {
        self.choice_map.get(&number).cloned()
    }

    pub fn max_depth(&self) -> i32 {
        self.max_depth
    }

    pub fn max_crossover_nodes(&self) -> i32 {
        self.max_crossover_nodes
    }

    pub fn max_mutation_nodes(&self) -> i32 {
        self.max_mutation_nodes
    }
}

#[derive(Debug)]
pub struct GpConfig {
    max_depth: i32,
    max_crossover_nodes: i32,
    max_mutation_nodes: i32,
    symbols: Vec<Rc<GpSymbol>>,
    symbol_map: HashMap<String, Rc<GpSymbol>>,
    start: Rc<GpRef>,
}

impl GpConfig {
    pub fn from_node(cfg: &Node) -> Result<Rc<Self>, GggpError> {
        let max_depth = node_int(cfg, "MaxDepth").max(1);
        let max_crossover_nodes = node_int(cfg, "MaxCrossoverNodes").max(1);
        let max_mutation_nodes = node_int(cfg, "MaxMutationNodes").max(1);
        let rules = cfg
            .child("RULES")
            .ok_or(GggpError::MissingRules)?;

        let mut symbols = Vec::new();
        let mut symbol_map = HashMap::new();
        let mut start_symbol: Option<Rc<GpSymbol>> = None;

        for symbol_node in rules.children() {
            let name = node_name(symbol_node);
            let symbol = build_symbol(
                symbol_node,
                max_depth,
                max_crossover_nodes,
                max_mutation_nodes,
            )?;
            if name.eq_ignore_ascii_case("START") {
                start_symbol = Some(Rc::clone(&symbol));
            }
            symbol_map.insert(name, Rc::clone(&symbol));
            symbols.push(symbol);
        }

        let start_symbol = start_symbol.ok_or(GggpError::MissingStart)?;
        let start_ref = start_symbol
            .choices()
            .get(0)
            .and_then(|choice| choice.refs().get(0))
            .cloned()
            .ok_or(GggpError::MissingStart)?;

        let mut config = GpConfig {
            max_depth,
            max_crossover_nodes,
            max_mutation_nodes,
            symbols,
            symbol_map,
            start: start_ref,
        };

        config.link_symbols()?;
        Ok(Rc::new(config))
    }

    pub fn symbols(&self) -> &[Rc<GpSymbol>] {
        &self.symbols
    }

    pub fn symbol(&self, name: &str) -> Option<Rc<GpSymbol>> {
        self.symbol_map.get(name).cloned()
    }

    pub fn start(&self) -> Rc<GpRef> {
        Rc::clone(&self.start)
    }

    pub fn tree_from_chromosome(&self, chromosome: &str) -> Result<GpTree, GggpError> {
        let genes = parse_chromosome(chromosome.trim())?;
        let mut gene_pos = 0usize;
        let error = Rc::new(Cell::new(false));
        build_tree_from_genes(self.start(), &genes, &mut gene_pos, None, None, error)
    }

    fn link_symbols(&mut self) -> Result<(), GggpError> {
        let original_symbols: Vec<Rc<GpSymbol>> = self.symbols.clone();
        for symbol in &original_symbols {
            for choice in symbol.choices() {
                for reference in choice.refs() {
                    if let Some(sym) = self.symbol(&reference.symbol_name) {
                        reference.set_symbol(&sym);
                    } else if reference.param_inc() > 0.0 {
                        let new_symbol = build_number_symbol(
                            &reference.symbol_name,
                            &reference.param_name,
                            reference.param_from,
                            reference.param_to,
                            reference.param_inc,
                            reference.param_optimize,
                            self.max_depth,
                            self.max_crossover_nodes,
                            self.max_mutation_nodes,
                        );
                        self.symbol_map
                            .insert(reference.symbol_name.clone(), Rc::clone(&new_symbol));
                        self.symbols.push(Rc::clone(&new_symbol));
                        reference.set_symbol(&new_symbol);
                    } else {
                        return Err(GggpError::MissingSymbol(reference.symbol_name.clone()));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GpTree {
    tree_type: TreeType,
    ref_: Rc<GpRef>,
    choice: Option<Rc<GpChoice>>,
    depth: i32,
    max_depth: i32,
    children: Vec<GpTree>,
    error: Rc<Cell<bool>>,
}

impl Clone for GpTree {
    fn clone(&self) -> Self {
        clone_tree(self, self.ref_(), None, None, Rc::new(Cell::new(false)))
    }
}

impl GpTree {
    pub fn tree_type(&self) -> TreeType {
        self.tree_type
    }

    pub fn ref_(&self) -> Rc<GpRef> {
        Rc::clone(&self.ref_)
    }

    pub fn choice(&self) -> Option<Rc<GpChoice>> {
        self.choice.clone()
    }

    pub fn depth(&self) -> i32 {
        self.depth
    }

    pub fn max_depth(&self) -> i32 {
        self.max_depth
    }

    pub fn children(&self) -> &[GpTree] {
        &self.children
    }

    pub fn tree_depth(&self) -> i32 {
        let mut max_depth = 1;
        for child in &self.children {
            max_depth = max_depth.max(1 + child.tree_depth());
        }
        max_depth
    }

    pub fn text(&self) -> String {
        match self.tree_type {
            TreeType::Empty => String::new(),
            TreeType::Choice => {
                let choice = match &self.choice {
                    Some(choice) => choice,
                    None => return String::new(),
                };
                let text = choice.text();
                if self.children.is_empty() {
                    return text.to_string();
                }
                let bytes = text.as_bytes();
                let mut result = String::new();
                let mut cursor: usize = 0;
                for (idx, reference) in choice.refs().iter().enumerate() {
                    let pos = reference.pos() as isize;
                    let len = reference.len() as isize;
                    if pos <= 0 || len < 0 {
                        continue;
                    }
                    let end_exclusive = pos.saturating_sub(2) as usize;
                    if end_exclusive >= cursor && end_exclusive <= bytes.len() {
                        result.push_str(&text[cursor..end_exclusive]);
                    }
                    if let Some(child) = self.children.get(idx) {
                        result.push_str(&child.text());
                    }
                    let next_cursor = (pos + len) as usize;
                    cursor = next_cursor.min(bytes.len());
                }
                if cursor <= bytes.len() {
                    result.push_str(&text[cursor..]);
                }
                result
            }
        }
    }

    pub fn chromosome(&self) -> String {
        match self.tree_type {
            TreeType::Empty => String::new(),
            TreeType::Choice => {
                let choice = match &self.choice {
                    Some(choice) => choice,
                    None => return String::new(),
                };
                let symbol = choice.symbol();
                let mut result = if symbol.choices().len() < 2 {
                    String::new()
                } else {
                    choice.number().to_string()
                };
                for child in &self.children {
                    let part = child.chromosome();
                    if !part.is_empty() {
                        if result.is_empty() {
                            result = part;
                        } else {
                            result.push('-');
                            result.push_str(&part);
                        }
                    }
                }
                result
            }
        }
    }

    pub fn indented_text(&self, indent: usize) -> String {
        match self.tree_type {
            TreeType::Empty => String::new(),
            TreeType::Choice => {
                let choice = match &self.choice {
                    Some(choice) => choice,
                    None => return String::new(),
                };
                let text = choice.text();
                if self.children.is_empty() {
                    return text.to_string();
                }
                let bytes = text.as_bytes();
                let mut result = String::new();
                let mut cursor: usize = 0;
                for (idx, reference) in choice.refs().iter().enumerate() {
                    let pos = reference.pos() as isize;
                    let len = reference.len() as isize;
                    if pos <= 0 || len < 0 {
                        continue;
                    }
                    let end_exclusive = pos.saturating_sub(2) as usize;
                    if end_exclusive >= cursor && end_exclusive <= bytes.len() {
                        result.push_str(&text[cursor..end_exclusive]);
                    }
                    let line_indent = current_line_indent(&result);
                    if let Some(child) = self.children.get(idx) {
                        result.push_str(&child.indented_text(line_indent));
                    }
                    let next_cursor = (pos + len) as usize;
                    cursor = next_cursor.min(bytes.len());
                }
                if cursor <= bytes.len() {
                    result.push_str(&text[cursor..]);
                }
                apply_indent(&result, indent)
            }
        }
    }

    pub fn matches(&self, other: &GpTree) -> bool {
        if self.tree_type != other.tree_type {
            return false;
        }
        if !Rc::ptr_eq(&self.ref_, &other.ref_) {
            return false;
        }
        match (&self.choice, &other.choice) {
            (Some(a), Some(b)) if !Rc::ptr_eq(a, b) => return false,
            (None, Some(_)) | (Some(_), None) => return false,
            _ => {}
        }
        if self.children.len() != other.children.len() {
            return false;
        }
        for (a, b) in self.children.iter().zip(other.children.iter()) {
            if !a.matches(b) {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone)]
pub struct GpIndividual {
    trees: Vec<GpTree>,
    has_fitness: bool,
    success: bool,
    fitness: f64,
    pub continuous_weights: Vec<f64>,
}

impl GpIndividual {
    pub fn new() -> Self {
        GpIndividual {
            trees: Vec::new(),
            has_fitness: false,
            success: false,
            fitness: -1.0,
            continuous_weights: Vec::new(),
        }
    }

    pub fn trees(&self) -> &[GpTree] {
        &self.trees
    }

    pub fn trees_mut(&mut self) -> &mut Vec<GpTree> {
        &mut self.trees
    }

    pub fn has_fitness(&self) -> bool {
        self.has_fitness
    }

    pub fn set_has_fitness(&mut self, value: bool) {
        self.has_fitness = value;
    }

    pub fn success(&self) -> bool {
        self.success
    }

    pub fn set_success(&mut self, value: bool) {
        self.success = value;
    }

    pub fn fitness(&self) -> f64 {
        self.fitness
    }

    pub fn set_fitness(&mut self, value: f64) {
        self.fitness = value;
    }

    pub fn random_trees(&mut self, configs: &[Rc<GpConfig>], rng: &mut impl Rng) {
        self.trees.clear();
        for cfg in configs {
            let start = cfg.start();
            let error = Rc::new(Cell::new(false));
            let tree = build_random_tree(start, None, None, error, rng);
            self.trees.push(tree);
        }
    }

    pub fn clone_from(&mut self, other: &GpIndividual) {
        self.trees.clear();
        for tree in &other.trees {
            let error = Rc::new(Cell::new(false));
            let cloned = clone_tree(tree, tree.ref_(), None, None, error);
            self.trees.push(cloned);
        }
        self.has_fitness = false;
        self.success = false;
        self.fitness = -1.0;
    }

    pub fn hash(&self) -> u32 {
        let mut text = String::new();
        for tree in &self.trees {
            text.push_str(&tree.chromosome());
            text.push(';');
        }
        str_hash_key(text.as_bytes())
    }

    pub fn size(&self) -> usize {
        let mut size = 4 + 1 + 1 + 8;
        for tree in &self.trees {
            let chromo = tree.chromosome();
            size += 4 + chromo.len();
        }
        size
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_i32(writer, self.trees.len() as i32)?;
        write_bool(writer, self.has_fitness)?;
        write_bool(writer, self.success)?;
        write_f64(writer, self.fitness)?;
        for tree in &self.trees {
            let chromo = tree.chromosome();
            write_i32(writer, chromo.len() as i32)?;
            writer.write_all(chromo.as_bytes())?;
        }
        Ok(())
    }

    pub fn read_from<R: Read>(reader: &mut R, configs: &[Rc<GpConfig>]) -> Result<Self, GggpError> {
        let count = read_i32(reader)?;
        if count < 0 {
            return Err(GggpError::InvalidData("negative tree count".to_string()));
        }
        let count = count as usize;
        if count != configs.len() {
            return Err(GggpError::InvalidData(format!(
                "tree count mismatch ({} != {})",
                count,
                configs.len()
            )));
        }
        let has_fitness = read_bool(reader)?;
        let success = read_bool(reader)?;
        let fitness = read_f64(reader)?;
        let mut trees = Vec::with_capacity(count);
        for i in 0..count {
            let len = read_i32(reader)?;
            if len < 0 {
                return Err(GggpError::InvalidData("negative chromosome length".to_string()));
            }
            let bytes = read_bytes(reader, len as usize)?;
            let chromosome = String::from_utf8(bytes)
                .map_err(|_| GggpError::InvalidData("invalid chromosome utf-8".to_string()))?;
            let genes = parse_chromosome(&chromosome)?;
            let mut gene_pos = 0usize;
            let error = Rc::new(Cell::new(false));
            let tree = build_tree_from_genes(
                configs[i].start(),
                &genes,
                &mut gene_pos,
                None,
                None,
                error,
            )?;
            trees.push(tree);
        }

        Ok(GpIndividual {
            trees,
            has_fitness,
            success,
            fitness,
            continuous_weights: Vec::new(),
        })
    }
}

pub struct Gggp {
    configs: Vec<Rc<GpConfig>>,
    population_size: usize,
    elite_size: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    optimize_direction: OptimizeDirection,
    individuals: Vec<GpIndividual>,
    fitness_history: HashMap<u32, FitnessHistoryRec>,
    rng: rand::rngs::ThreadRng,
    on_get_fitness: Option<Box<dyn FnMut(&GpIndividual) -> Option<f64>>>,
    on_active_change: Option<Box<dyn FnMut(&GpIndividual, bool)>>,
    on_check_stop_pause: Option<Box<dyn FnMut(&mut bool, &mut bool)>>,
}

impl Gggp {
    pub fn new() -> Self {
        Gggp {
            configs: Vec::new(),
            population_size: 0,
            elite_size: 0,
            crossover_probability: 0.0,
            mutation_probability: 0.0,
            optimize_direction: OptimizeDirection::Maximize,
            individuals: Vec::new(),
            fitness_history: HashMap::new(),
            rng: rand::thread_rng(),
            on_get_fitness: None,
            on_active_change: None,
            on_check_stop_pause: None,
        }
    }

    pub fn set_on_get_fitness<F>(&mut self, func: F)
    where
        F: FnMut(&GpIndividual) -> Option<f64> + 'static,
    {
        self.on_get_fitness = Some(Box::new(func));
    }

    pub fn set_on_active_change<F>(&mut self, func: F)
    where
        F: FnMut(&GpIndividual, bool) + 'static,
    {
        self.on_active_change = Some(Box::new(func));
    }

    pub fn set_on_check_stop_pause<F>(&mut self, func: F)
    where
        F: FnMut(&mut bool, &mut bool) + 'static,
    {
        self.on_check_stop_pause = Some(Box::new(func));
    }

    pub fn configs(&self) -> &[Rc<GpConfig>] {
        &self.configs
    }

    pub fn individuals(&self) -> &[GpIndividual] {
        &self.individuals
    }

    pub fn individuals_mut(&mut self) -> &mut Vec<GpIndividual> {
        &mut self.individuals
    }

    pub fn init_from_nodes(
        &mut self,
        cfgs: &[Node],
        population_size: usize,
        elite_size: usize,
        crossover_probability: f64,
        mutation_probability: f64,
    ) -> Result<(), GggpError> {
        if elite_size > population_size {
            return Err(GggpError::InvalidConfig("elite size > population size"));
        }
        if self.on_get_fitness.is_none() {
            return Err(GggpError::InvalidConfig("OnGetFitness not assigned"));
        }

        self.population_size = population_size;
        self.elite_size = elite_size;
        self.crossover_probability = crossover_probability;
        self.mutation_probability = mutation_probability;

        self.configs.clear();
        for cfg in cfgs {
            let gcfg = GpConfig::from_node(cfg)?;
            self.configs.push(gcfg);
        }
        self.reset();
        Ok(())
    }

    pub fn reset(&mut self) {
        self.fitness_history.clear();
        self.individuals.clear();
        self.init_population();
    }

    fn init_population(&mut self) {
        let mut hashes = HashSet::new();
        let mut attempts = 0usize;
        let max_attempts = self.population_size.saturating_mul(10).max(1);
        while self.individuals.len() < self.population_size && attempts < max_attempts {
            let mut ind = GpIndividual::new();
            ind.random_trees(&self.configs, &mut self.rng);
            let h = ind.hash();
            if !hashes.contains(&h) && !self.fitness_history.contains_key(&h) {
                hashes.insert(h);
                self.individuals.push(ind);
                attempts = 0;
            } else {
                attempts += 1;
            }
        }
    }

    pub fn step(&mut self) {
        let mut i = 0usize;
        while i < self.individuals.len() {
            let fitness = (self.on_get_fitness.as_mut().unwrap())(&self.individuals[i]);
            match fitness {
                Some(value) => {
                    self.individuals[i].fitness = value;
                    self.individuals[i].has_fitness = true;
                    self.individuals[i].success = true;
                    i += 1;
                }
                None => {
                    self.individuals.remove(i);
                }
            }
        }

        self.sort_population();
        let old_population = self.individuals.clone();
        self.individuals.clear();

        for i in 0..self.elite_size.min(old_population.len()) {
            let mut ind = GpIndividual::new();
            ind.clone_from(&old_population[i]);
            self.individuals.push(ind);
        }

        while self.individuals.len() < self.population_size {
            let ind1 = select_tournament(&old_population, self.optimize_direction, &mut self.rng);
            if self.rng.gen::<f64>() < self.crossover_probability {
                let mut ind2 = select_tournament(&old_population, self.optimize_direction, &mut self.rng);
                while ind2.hash() == ind1.hash() {
                    ind2 = select_tournament(&old_population, self.optimize_direction, &mut self.rng);
                }

                let mut new1 = GpIndividual::new();
                let mut new2 = GpIndividual::new();
                new1.clone_from(&ind1);
                new2.clone_from(&ind2);
                self.crossover(&mut new1, &mut new2);
                if self.rng.gen::<f64>() < self.mutation_probability {
                    self.mutation(&mut new1);
                }
                if self.rng.gen::<f64>() < self.mutation_probability {
                    self.mutation(&mut new2);
                }
                self.individuals.push(new1);
                if self.individuals.len() < self.population_size {
                    self.individuals.push(new2);
                }
            } else {
                let mut new_ind = GpIndividual::new();
                new_ind.clone_from(&ind1);
                if self.rng.gen::<f64>() < self.mutation_probability {
                    self.mutation(&mut new_ind);
                }
                self.individuals.push(new_ind);
            }
        }
    }

    pub fn step_ssga(&mut self) {
        self.calc_fitnesses();
        if self.should_stop() {
            return;
        }

        let count = self.individuals.len();
        if count > 1 {
            let mut matrix = SelectionMatrix::new(count);
            loop {
                let (idx1, idx2) = match matrix.select_pair(
                    &self.individuals,
                    self.optimize_direction,
                    &mut self.rng,
                ) {
                    Some(pair) => pair,
                    None => break,
                };

                let mut new1 = GpIndividual::new();
                let mut new2 = GpIndividual::new();
                new1.clone_from(&self.individuals[idx1]);
                new2.clone_from(&self.individuals[idx2]);

                let cr = self.crossover(&mut new1, &mut new2);
                let m1 = self.mutation(&mut new1);
                let m2 = self.mutation(&mut new2);

                if !cr {
                    if !m1 {
                        new1.trees.clear();
                    }
                    if !m2 {
                        new2.trees.clear();
                    }
                }

                if !new1.trees.is_empty() {
                    let h = new1.hash();
                    if !self.fitness_history.contains_key(&h) {
                        self.individuals.push(new1);
                    }
                }
                if !new2.trees.is_empty() {
                    let h = new2.hash();
                    if !self.fitness_history.contains_key(&h) {
                        self.individuals.push(new2);
                    }
                }

                if self.should_stop() {
                    return;
                }
            }

            self.calc_fitnesses();
        }

        self.trim_or_extend_population();
    }

    fn calc_fitnesses(&mut self) {
        let mut i = 0usize;
        while i < self.individuals.len() {
            if !self.individuals[i].has_fitness {
                let h = self.individuals[i].hash();
                if let Some(rec) = self.fitness_history.get(&h) {
                    if rec.success {
                        self.individuals[i].fitness = rec.fitness;
                        self.individuals[i].has_fitness = true;
                        self.individuals[i].success = true;
                        if let Some(callback) = self.on_active_change.as_mut() {
                            callback(&self.individuals[i], true);
                        }
                        i += 1;
                    } else {
                        self.individuals.remove(i);
                    }
                } else {
                    let res = (self.on_get_fitness.as_mut().unwrap())(&self.individuals[i]);
                    match res {
                        Some(value) => {
                            self.individuals[i].fitness = value;
                            self.individuals[i].has_fitness = true;
                            self.individuals[i].success = true;
                            self.fitness_history.insert(
                                h,
                                FitnessHistoryRec {
                                    success: true,
                                    fitness: value,
                                },
                            );
                            if let Some(callback) = self.on_active_change.as_mut() {
                                callback(&self.individuals[i], true);
                            }
                            i += 1;
                        }
                        None => {
                            self.fitness_history.insert(
                                h,
                                FitnessHistoryRec {
                                    success: false,
                                    fitness: 0.0,
                                },
                            );
                            self.individuals.remove(i);
                        }
                    }
                }
            } else {
                i += 1;
            }

            if self.should_stop() {
                return;
            }
        }

        self.sort_population();
    }

    fn sort_population(&mut self) {
        match self.optimize_direction {
            OptimizeDirection::Maximize => {
                self.individuals.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            }
            OptimizeDirection::Minimize => {
                self.individuals.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
            }
        }
    }

    fn trim_or_extend_population(&mut self) {
        let count = self.individuals.len();
        if count > self.population_size {
            match self.optimize_direction {
                OptimizeDirection::Maximize => {
                    while self.individuals.len() > self.population_size {
                        if let Some(callback) = self.on_active_change.as_mut() {
                            if let Some(ind) = self.individuals.first() {
                                callback(ind, false);
                            }
                        }
                        self.individuals.remove(0);
                    }
                }
                OptimizeDirection::Minimize => {
                    while self.individuals.len() > self.population_size {
                        if let Some(callback) = self.on_active_change.as_mut() {
                            if let Some(ind) = self.individuals.last() {
                                callback(ind, false);
                            }
                        }
                        self.individuals.pop();
                    }
                }
            }
        } else if count < self.population_size {
            self.init_population();
        }
    }

    fn should_stop(&mut self) -> bool {
        if let Some(callback) = self.on_check_stop_pause.as_mut() {
            let mut stop = false;
            let mut pause = false;
            callback(&mut stop, &mut pause);
            return stop;
        }
        false
    }

    pub fn crossover(&mut self, ind1: &mut GpIndividual, ind2: &mut GpIndividual) -> bool {
        crossover_with_rng(&mut self.rng, self.crossover_probability, ind1, ind2)
    }

    pub fn mutation(&mut self, ind: &mut GpIndividual) -> bool {
        mutation_with_rng(&mut self.rng, self.mutation_probability, ind)
    }

    pub fn size(&self) -> usize {
        individuals_size(&self.individuals) + fitness_history_size(&self.fitness_history)
    }

    pub fn save_to_writer<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_i32(writer, self.individuals.len() as i32)?;
        for ind in &self.individuals {
            ind.write_to(writer)?;
        }
        write_fitness_history(writer, &self.fitness_history)?;
        Ok(())
    }

    pub fn save_to_vec(&self) -> io::Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.size());
        self.save_to_writer(&mut out)?;
        Ok(out)
    }

    pub fn load_from_reader<R: Read>(&mut self, reader: &mut R) -> Result<(), GggpError> {
        if self.configs.is_empty() {
            return Err(GggpError::InvalidConfig("configs not initialized"));
        }
        let count = read_i32(reader)?;
        if count < 0 {
            return Err(GggpError::InvalidData("negative individuals count".to_string()));
        }
        let count = count as usize;
        self.individuals.clear();
        for _ in 0..count {
            let ind = GpIndividual::read_from(reader, &self.configs)?;
            self.individuals.push(ind);
        }
        self.fitness_history = read_fitness_history(reader)?;
        Ok(())
    }

    pub fn load_from_bytes(&mut self, data: &[u8]) -> Result<(), GggpError> {
        let mut cursor = io::Cursor::new(data);
        self.load_from_reader(&mut cursor)
    }
}

pub fn crossover_individuals<R: Rng>(
    rng: &mut R,
    ind1: &mut GpIndividual,
    ind2: &mut GpIndividual,
) -> bool {
    crossover_with_rng(rng, 1.0, ind1, ind2)
}

pub fn mutate_individual<R: Rng>(rng: &mut R, ind: &mut GpIndividual) -> bool {
    mutation_with_rng(rng, 1.0, ind)
}

fn crossover_with_rng<R: Rng>(
    rng: &mut R,
    probability: f64,
    ind1: &mut GpIndividual,
    ind2: &mut GpIndividual,
) -> bool {
    if ind1.trees.len() != ind2.trees.len() {
        return false;
    }
    let mut result = false;

    for ti in 0..ind1.trees.len() {
        if rng.gen::<f64>() >= probability {
            continue;
        }
        let mut tree1 = ind1.trees[ti].clone();
        let mut tree2 = ind2.trees[ti].clone();
        let mut modified = false;
        let max_nodes = tree1.ref_().symbol().max_crossover_nodes();
        let mut todo = ((rng.gen::<f64>() * max_nodes as f64).ceil() as i32).max(1);

        while todo > 0 {
            let mut done = false;
            let mut nt1 = Vec::new();
            let mut nt2 = Vec::new();
            let mut depth1 = 0;
            let mut depth2 = 0;
            collect_non_terminals(&tree1, &mut Vec::new(), &mut nt1, &mut depth1);
            collect_non_terminals(&tree2, &mut Vec::new(), &mut nt2, &mut depth2);

            while !nt1.is_empty() && !done {
                let idx1 = rng.gen_range(0..nt1.len());
                let cn1 = nt1[idx1].clone();
                let cn1_text = cn1.text.clone();

                let parent_choice = cn1.reference.choice();
                let parent_symbol = parent_choice.symbol();
                let cn1_idx = match parent_choice.index_of_ref_number(cn1.reference.number()) {
                    Some(idx) => idx,
                    None => {
                        nt1.remove(idx1);
                        continue;
                    }
                };

                let mut candidates = Vec::new();
                for choice in parent_symbol.choices() {
                    if choice.refs().len() != parent_choice.refs().len() {
                        continue;
                    }
                    let mut ok = true;
                    for (idx, ref_) in choice.refs().iter().enumerate() {
                        if idx == cn1_idx {
                            continue;
                        }
                        let sym_a = ref_.symbol();
                        let sym_b = parent_choice.refs()[idx].symbol();
                        if !Rc::ptr_eq(&sym_a, &sym_b) {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        candidates.push(Rc::clone(choice));
                    }
                }

                let mut symbols = Vec::new();
                for cand in &candidates {
                    let sym = cand.refs()[cn1_idx].symbol();
                    if !symbols.iter().any(|s: &Rc<GpSymbol>| Rc::ptr_eq(s, &sym)) {
                        symbols.push(sym);
                    }
                }

                while !symbols.is_empty() && !done {
                    let cs_idx = rng.gen_range(0..symbols.len());
                    let cs = symbols[cs_idx].clone();

                    let mut pn: Vec<NodeInfo> = nt2
                        .iter()
                        .filter(|node| Rc::ptr_eq(&node.reference.symbol(), &cs) && node.text != cn1_text)
                        .cloned()
                        .collect();

                    while !pn.is_empty() && !done {
                        let idx2 = rng.gen_range(0..pn.len());
                        let cn2 = pn[idx2].clone();

                        let tmp1 = replace_tree(&tree1, &cn1.path, &cn2.node, rng);
                        if tmp1.is_none() {
                            pn.remove(idx2);
                            continue;
                        }
                        let tmp2 = replace_tree(&tree2, &cn2.path, &cn1.node, rng);
                        if tmp2.is_none() {
                            pn.remove(idx2);
                            continue;
                        }

                        tree1 = tmp1.unwrap();
                        tree2 = tmp2.unwrap();
                        modified = true;
                        todo -= 1;
                        done = true;
                        result = true;
                    }

                    if !done {
                        symbols.remove(cs_idx);
                    }
                }

                if !done {
                    nt1.remove(idx1);
                }
            }

            if !done {
                break;
            }
        }

        if modified {
            ind1.trees[ti] = tree1;
            ind2.trees[ti] = tree2;
        }
    }

    result
}

fn mutation_with_rng<R: Rng>(
    rng: &mut R,
    probability: f64,
    ind: &mut GpIndividual,
) -> bool {
    let mut result = false;
    for ti in 0..ind.trees.len() {
        if rng.gen::<f64>() >= probability {
            continue;
        }

        let mut tree = ind.trees[ti].clone();
        let mut modified = false;
        let max_nodes = tree.ref_().symbol().max_mutation_nodes();
        let mut todo = ((rng.gen::<f64>() * max_nodes as f64).ceil() as i32).max(1);

        while todo > 0 {
            let mut done = false;
            let mut nt = Vec::new();
            let mut depth = 0;
            collect_non_terminals(&tree, &mut Vec::new(), &mut nt, &mut depth);

            while !nt.is_empty() && !done {
                let idx = rng.gen_range(0..nt.len());
                let mn = nt[idx].clone();
                let parent_choice = mn.reference.choice();
                let parent_symbol = parent_choice.symbol();
                let mn_idx = match parent_choice.index_of_ref_number(mn.reference.number()) {
                    Some(idx) => idx,
                    None => {
                        nt.remove(idx);
                        continue;
                    }
                };

                let mut candidates = Vec::new();
                for choice in parent_symbol.choices() {
                    if choice.refs().len() != parent_choice.refs().len() {
                        continue;
                    }
                    let mut ok = true;
                    for (pos, ref_) in choice.refs().iter().enumerate() {
                        if pos == mn_idx {
                            continue;
                        }
                        let sym_a = ref_.symbol();
                        let sym_b = parent_choice.refs()[pos].symbol();
                        if !Rc::ptr_eq(&sym_a, &sym_b) {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        candidates.push(Rc::clone(choice));
                    }
                }

                let mut symbols = Vec::new();
                for cand in &candidates {
                    let sym = cand.refs()[mn_idx].symbol();
                    if !symbols.iter().any(|s: &Rc<GpSymbol>| Rc::ptr_eq(s, &sym)) {
                        symbols.push(sym);
                    }
                }

                while !symbols.is_empty() && !done {
                    let ms_idx = rng.gen_range(0..symbols.len());
                    let ms = symbols[ms_idx].clone();

                    let tmp = replace_tree_random(&tree, &mn.path, ms.clone(), rng);
                    if let Some(new_tree) = tmp {
                        tree = new_tree;
                        modified = true;
                        todo -= 1;
                        done = true;
                        result = true;
                    } else {
                        symbols.remove(ms_idx);
                    }
                }

                if !done {
                    nt.remove(idx);
                }
            }

            if !done {
                break;
            }
        }

        if modified {
            ind.trees[ti] = tree;
        }
    }
    result
}

#[derive(Debug, Clone)]
struct NodeInfo {
    path: Vec<usize>,
    reference: Rc<GpRef>,
    node: GpTree,
    text: String,
    depth: i32,
}

fn collect_non_terminals(
    current: &GpTree,
    path: &mut Vec<usize>,
    output: &mut Vec<NodeInfo>,
    max_depth: &mut i32,
) {
    for (idx, child) in current.children.iter().enumerate() {
        *max_depth = (*max_depth).max(child.depth());
        path.push(idx);
        if let Some(choice) = child.choice() {
            if !choice.symbol().terminal() {
                output.push(NodeInfo {
                    path: path.clone(),
                    reference: child.ref_(),
                    node: clone_tree(child, child.ref_(), None, None, Rc::new(Cell::new(false))),
                    text: child.text(),
                    depth: child.depth(),
                });
            }
        }
        collect_non_terminals(child, path, output, max_depth);
        path.pop();
    }
}

fn build_symbol(
    cfg: &Node,
    max_depth: i32,
    max_crossover_nodes: i32,
    max_mutation_nodes: i32,
) -> Result<Rc<GpSymbol>, GggpError> {
    let name = node_name(cfg);
    let min_depth = node_int(cfg, "Length");
    let choices_node = cfg
        .child("CHOICES")
        .ok_or_else(|| GggpError::InvalidConfig("symbol without choices"))?;

    let symbol = Rc::new_cyclic(|weak_symbol| {
        let mut choices = Vec::new();
        for choice_node in choices_node.children() {
            let choice = build_choice(weak_symbol.clone(), choice_node);
            choices.push(choice);
        }

        let terminal = choices.len() == 1
            && choices
                .get(0)
                .map(|choice| choice.refs().is_empty())
                .unwrap_or(false);

        let mut choice_map = HashMap::new();
        for choice in &choices {
            choice_map.insert(choice.number(), Rc::clone(choice));
        }

        GpSymbol {
            name: name.clone(),
            min_depth,
            terminal,
            choices,
            choice_map,
            max_depth,
            max_crossover_nodes,
            max_mutation_nodes,
        }
    });

    Ok(symbol)
}

fn build_choice(symbol: Weak<GpSymbol>, cfg: &Node) -> Rc<GpChoice> {
    Rc::new_cyclic(|weak_choice| {
        let number = node_name(cfg).parse::<i32>().unwrap_or(0);
        let text = node_str(cfg, "Text");
        let min_depth = node_int(cfg, "Length");

        let mut refs = Vec::new();
        let mut ref_map = HashMap::new();
        if let Some(refs_node) = cfg.child("REFS") {
            for group in refs_node.children() {
                let symbol_name = node_name(group);
                for ref_node in group.children() {
                    let reference = build_ref(weak_choice.clone(), &symbol_name, ref_node);
                    ref_map.insert(reference.number(), Rc::clone(&reference));
                    refs.push(reference);
                }
            }
        }
        refs.sort_by_key(|r| r.number());

        GpChoice {
            symbol: symbol.clone(),
            number,
            text,
            min_depth,
            refs,
            ref_map,
        }
    })
}

fn build_ref(choice: Weak<GpChoice>, symbol_name: &str, cfg: &Node) -> Rc<GpRef> {
    let number = node_name(cfg).parse::<i32>().unwrap_or(0);
    let pos = node_int(cfg, "Pos");
    let len = node_int(cfg, "Len");
    let mut param_max_depth = -1;
    if let Some(params) = cfg.child("PARAMS") {
        let max_depth_str = node_str(params, "maxdepth");
        param_max_depth = max_depth_str.parse::<i32>().unwrap_or(-1);
    }
    let number_params = is_ref_to_number(cfg);
    let (param_name, param_from, param_to, param_inc, param_optimize) = number_params
        .map(|params| {
            (
                params.name,
                params.from,
                params.to,
                params.inc,
                params.optimize,
            )
        })
        .unwrap_or((String::new(), 0.0, 0.0, 0.0, false));

    Rc::new(GpRef {
        choice,
        number,
        symbol_name: symbol_name.to_string(),
        symbol: RefCell::new(None),
        pos,
        len,
        param_max_depth,
        param_name,
        param_from,
        param_to,
        param_inc,
        param_optimize,
    })
}

fn build_number_symbol(
    symbol_name: &str,
    number_name: &str,
    from: f64,
    to: f64,
    inc: f64,
    optimize: bool,
    max_depth: i32,
    max_crossover_nodes: i32,
    max_mutation_nodes: i32,
) -> Rc<GpSymbol> {
    Rc::new_cyclic(|weak_symbol| {
        let mut choices = Vec::new();
        let mut choice_map = HashMap::new();
        let mut idx = 0;
        let mut value = from;
        while value <= to {
            let choice = build_number_choice(
                weak_symbol.clone(),
                idx,
                value,
                number_name,
                from,
                to,
                inc,
                optimize,
            );
            choice_map.insert(choice.number(), Rc::clone(&choice));
            choices.push(choice);
            value += inc;
            idx += 1;
        }

        GpSymbol {
            name: symbol_name.to_string(),
            min_depth: 1,
            terminal: choices.len() == 1,
            choices,
            choice_map,
            max_depth,
            max_crossover_nodes,
            max_mutation_nodes,
        }
    })
}

fn build_number_choice(
    symbol: Weak<GpSymbol>,
    index: i32,
    value: f64,
    number_name: &str,
    from: f64,
    to: f64,
    inc: f64,
    optimize: bool,
) -> Rc<GpChoice> {
    Rc::new_cyclic(|_weak_choice| {
        let value_text = format_float_raw(value);
        let from_text = format_float_raw(from);
        let to_text = format_float_raw(to);
        let inc_text = format_float_raw(inc);
        let text = if optimize {
            if (inc - 1.0).abs() < f64::EPSILON {
                format!(
                    "Param(%ParamId%, {}{{{}{}-{}}})",
                    value_text, number_name, from_text, to_text
                )
            } else {
                format!(
                    "Param(%ParamId%, {}{{{}{}-{}`{}}})",
                    value_text, number_name, from_text, to_text, inc_text
                )
            }
        } else if (inc - 1.0).abs() < f64::EPSILON {
            format!("{}{{{}{}-{}}}", value_text, number_name, from_text, to_text)
        } else {
            format!(
                "{}{{{}{}-{}`{}}}",
                value_text, number_name, from_text, to_text, inc_text
            )
        };

        GpChoice {
            symbol: symbol.clone(),
            number: index,
            text,
            min_depth: 1,
            refs: Vec::new(),
            ref_map: HashMap::new(),
        }
    })
}

fn select_tournament(
    population: &[GpIndividual],
    direction: OptimizeDirection,
    rng: &mut impl Rng,
) -> GpIndividual {
    let count = population.len();
    if count == 0 {
        return GpIndividual::new();
    }
    if count == 1 {
        return population[0].clone();
    }
    let idx1 = rng.gen_range(0..count);
    let mut idx2 = rng.gen_range(0..count);
    while idx2 == idx1 {
        idx2 = rng.gen_range(0..count);
    }
    let ind1 = &population[idx1];
    let ind2 = &population[idx2];
    let selected = match direction {
        OptimizeDirection::Minimize => {
            if ind1.fitness < ind2.fitness {
                ind1
            } else {
                ind2
            }
        }
        OptimizeDirection::Maximize => {
            if ind1.fitness > ind2.fitness {
                ind1
            } else {
                ind2
            }
        }
    };
    let mut result = GpIndividual::new();
    result.clone_from(selected);
    result
}

#[derive(Debug)]
struct SelectionMatrix {
    available: Vec<Vec<bool>>,
    counts: Vec<usize>,
    total: usize,
}

impl SelectionMatrix {
    fn new(count: usize) -> Self {
        let mut available = Vec::new();
        let mut counts = Vec::new();
        for i in 0..count {
            let mut row = vec![false; count];
            row[i] = true;
            available.push(row);
            counts.push(count.saturating_sub(1));
        }
        SelectionMatrix {
            available,
            counts,
            total: count,
        }
    }

    fn select_pair(
        &mut self,
        individuals: &[GpIndividual],
        direction: OptimizeDirection,
        rng: &mut impl Rng,
    ) -> Option<(usize, usize)> {
        if self.total < 1 {
            return None;
        }
        let first = if self.total == 1 {
            self.first_available_index(0)?
        } else {
            let r1 = rng.gen_range(0..self.total);
            let mut r2 = rng.gen_range(0..self.total);
            while r2 == r1 {
                r2 = rng.gen_range(0..self.total);
            }
            let i1 = self.first_available_index(r1)?;
            let i2 = self.first_available_index(r2)?;
            let better = match direction {
                OptimizeDirection::Maximize => {
                    if individuals[i1].fitness > individuals[i2].fitness {
                        i1
                    } else {
                        i2
                    }
                }
                OptimizeDirection::Minimize => {
                    if individuals[i1].fitness < individuals[i2].fitness {
                        i1
                    } else {
                        i2
                    }
                }
            };
            better
        };

        let second_count = self.counts[first];
        if second_count == 0 {
            return None;
        }
        let second = if second_count == 1 {
            self.second_available_index(first, 0)?
        } else {
            let r1 = rng.gen_range(0..second_count);
            let mut r2 = rng.gen_range(0..second_count);
            while r2 == r1 {
                r2 = rng.gen_range(0..second_count);
            }
            let i1 = self.second_available_index(first, r1)?;
            let i2 = self.second_available_index(first, r2)?;
            let better = match direction {
                OptimizeDirection::Maximize => {
                    if individuals[i1].fitness > individuals[i2].fitness {
                        i1
                    } else {
                        i2
                    }
                }
                OptimizeDirection::Minimize => {
                    if individuals[i1].fitness < individuals[i2].fitness {
                        i1
                    } else {
                        i2
                    }
                }
            };
            better
        };

        self.mark_used(first, second);
        Some((first, second))
    }

    fn first_available_index(&self, mut num: usize) -> Option<usize> {
        for (idx, count) in self.counts.iter().enumerate() {
            if *count > 0 {
                if num == 0 {
                    return Some(idx);
                }
                num = num.saturating_sub(1);
            }
        }
        None
    }

    fn second_available_index(&self, row: usize, mut num: usize) -> Option<usize> {
        for (idx, used) in self.available[row].iter().enumerate() {
            if !*used {
                if num == 0 {
                    return Some(idx);
                }
                num = num.saturating_sub(1);
            }
        }
        None
    }

    fn mark_used(&mut self, i: usize, j: usize) {
        if !self.available[i][j] {
            self.available[i][j] = true;
            self.counts[i] = self.counts[i].saturating_sub(1);
            if self.counts[i] == 0 {
                self.total = self.total.saturating_sub(1);
            }
        }
        if !self.available[j][i] {
            self.available[j][i] = true;
            self.counts[j] = self.counts[j].saturating_sub(1);
            if self.counts[j] == 0 {
                self.total = self.total.saturating_sub(1);
            }
        }
    }
}

fn build_random_tree(
    reference: Rc<GpRef>,
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    error: Rc<Cell<bool>>,
    rng: &mut impl Rng,
) -> GpTree {
    if reference.param_max_depth() == 0 {
        return build_node(reference, None, TreeType::Empty, parent_depth, parent_max_depth, error);
    }

    let (depth, max_depth) = calc_depth_max(parent_depth, parent_max_depth, &reference);
    let symbol = reference.symbol();
    let mut valid_choices = Vec::new();
    for choice in symbol.choices() {
        if depth + choice.min_depth() <= max_depth {
            valid_choices.push(Rc::clone(choice));
        }
    }
    if valid_choices.is_empty() {
        let node = build_node(reference, None, TreeType::Empty, parent_depth, parent_max_depth, error);
        node.error.set(true);
        return node;
    }

    let choice = valid_choices[rng.gen_range(0..valid_choices.len())].clone();
    let mut node = build_node(
        reference,
        Some(choice.clone()),
        TreeType::Choice,
        parent_depth,
        parent_max_depth,
        error,
    );
    for ref_ in choice.refs() {
        let child = build_random_tree(Rc::clone(ref_), Some(node.depth), Some(node.max_depth), Rc::clone(&node.error), rng);
        node.children.push(child);
    }
    node
}

fn build_node(
    reference: Rc<GpRef>,
    choice: Option<Rc<GpChoice>>,
    tree_type: TreeType,
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    error: Rc<Cell<bool>>,
) -> GpTree {
    let (depth, max_depth) = calc_depth_max(parent_depth, parent_max_depth, &reference);
    GpTree {
        tree_type,
        ref_: reference,
        choice,
        depth,
        max_depth,
        children: Vec::new(),
        error,
    }
}

fn calc_depth_max(
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    reference: &GpRef,
) -> (i32, i32) {
    let depth = parent_depth.map(|d| d + 1).unwrap_or(0);
    let mut max_depth = parent_max_depth
        .unwrap_or_else(|| reference.symbol().max_depth());
    if reference.param_max_depth() >= 0 {
        max_depth = max_depth.min(depth + reference.param_max_depth());
    }
    (depth, max_depth)
}

fn clone_tree(
    from: &GpTree,
    reference: Rc<GpRef>,
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    error: Rc<Cell<bool>>,
) -> GpTree {
    let mut node = build_node(
        reference,
        from.choice.clone(),
        from.tree_type,
        parent_depth,
        parent_max_depth,
        error,
    );
    for child in &from.children {
        let child_clone = clone_tree(
            child,
            child.ref_(),
            Some(node.depth),
            Some(node.max_depth),
            Rc::clone(&node.error),
        );
        node.children.push(child_clone);
    }
    node
}

fn parse_chromosome(chromosome: &str) -> Result<Vec<i32>, GggpError> {
    if chromosome.is_empty() {
        return Ok(Vec::new());
    }
    chromosome
        .split('-')
        .map(|part| {
            part.parse::<i32>().map_err(|_| {
                GggpError::InvalidData(format!("invalid gene '{}'", part))
            })
        })
        .collect()
}

fn build_tree_from_genes(
    reference: Rc<GpRef>,
    genes: &[i32],
    gene_pos: &mut usize,
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    error: Rc<Cell<bool>>,
) -> Result<GpTree, GggpError> {
    if reference.param_max_depth() == 0 {
        return Ok(build_node(
            reference,
            None,
            TreeType::Empty,
            parent_depth,
            parent_max_depth,
            error,
        ));
    }

    let symbol = reference.symbol();
    if symbol.choices().is_empty() {
        error.set(true);
        return Err(GggpError::InvalidConfig("symbol has no rules"));
    }

    let choice = if symbol.choices().len() == 1 {
        Rc::clone(&symbol.choices()[0])
    } else {
        if *gene_pos >= genes.len() {
            error.set(true);
            return Err(GggpError::InvalidData(
                "chromosome ended before tree was built".to_string(),
            ));
        }
        let number = genes[*gene_pos];
        *gene_pos += 1;
        symbol.choice_by_number(number).ok_or_else(|| {
            error.set(true);
            GggpError::InvalidChoice(format!(
                "symbol {} has no choice {}",
                symbol.name(),
                number
            ))
        })?
    };

    let mut node = build_node(
        reference,
        Some(choice.clone()),
        TreeType::Choice,
        parent_depth,
        parent_max_depth,
        error,
    );
    for ref_ in choice.refs() {
        let child = build_tree_from_genes(
            Rc::clone(ref_),
            genes,
            gene_pos,
            Some(node.depth),
            Some(node.max_depth),
            Rc::clone(&node.error),
        )?;
        node.children.push(child);
    }
    Ok(node)
}

fn replace_tree(
    from: &GpTree,
    path: &[usize],
    new_node: &GpTree,
    rng: &mut impl Rng,
) -> Option<GpTree> {
    if path.is_empty() {
        return None;
    }
    let error = Rc::new(Cell::new(false));
    let tree = replace_tree_internal(
        from,
        path,
        new_node,
        rng,
        None,
        None,
        Rc::clone(&error),
    );
    if error.get() {
        None
    } else {
        Some(tree)
    }
}

fn replace_tree_internal(
    from: &GpTree,
    path: &[usize],
    new_node: &GpTree,
    rng: &mut impl Rng,
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    error: Rc<Cell<bool>>,
) -> GpTree {
    if path.is_empty() {
        return clone_tree(from, from.ref_(), parent_depth, parent_max_depth, error);
    }
    let (depth, max_depth) = calc_depth_max(parent_depth, parent_max_depth, &from.ref_());
    if path.len() == 1 {
        let replace_index = path[0];
        let mut symbols = Vec::new();
        let mut deps = Vec::new();
        for (idx, child) in from.children.iter().enumerate() {
            if idx == replace_index {
                symbols.push(new_node.ref_().symbol());
                deps.push(new_node.tree_depth());
            } else {
                symbols.push(child.ref_().symbol());
                deps.push(child.tree_depth());
            }
        }

        let parent_symbol = from.ref_().symbol();
        let choice = select_choice(&parent_symbol, from.choice.as_ref(), &symbols, &deps, depth, max_depth, rng);
        if let Some(choice) = choice {
            let mut node = build_node(
                from.ref_(),
                Some(choice.clone()),
                TreeType::Choice,
                parent_depth,
                parent_max_depth,
                error,
            );
            for (idx, ref_) in choice.refs().iter().enumerate() {
                if idx == replace_index {
                    let child = clone_tree(new_node, Rc::clone(ref_), Some(node.depth), Some(node.max_depth), Rc::clone(&node.error));
                    node.children.push(child);
                } else {
                    if let Some(old_child) = from.children.get(idx) {
                        let child = clone_tree(old_child, Rc::clone(ref_), Some(node.depth), Some(node.max_depth), Rc::clone(&node.error));
                        node.children.push(child);
                    }
                }
            }
            node
        } else {
            let node = build_node(from.ref_(), None, TreeType::Empty, parent_depth, parent_max_depth, error);
            node.error.set(true);
            node
        }
    } else {
        let mut node = build_node(
            from.ref_(),
            from.choice.clone(),
            from.tree_type,
            parent_depth,
            parent_max_depth,
            error,
        );
        for (idx, child) in from.children.iter().enumerate() {
            if idx == path[0] {
                let sub = replace_tree_internal(
                    child,
                    &path[1..],
                    new_node,
                    rng,
                    Some(node.depth),
                    Some(node.max_depth),
                    Rc::clone(&node.error),
                );
                node.children.push(sub);
            } else {
                let clone = clone_tree(
                    child,
                    child.ref_(),
                    Some(node.depth),
                    Some(node.max_depth),
                    Rc::clone(&node.error),
                );
                node.children.push(clone);
            }
        }
        node
    }
}

fn replace_tree_random(
    from: &GpTree,
    path: &[usize],
    new_symbol: Rc<GpSymbol>,
    rng: &mut impl Rng,
) -> Option<GpTree> {
    if path.is_empty() {
        return None;
    }
    let error = Rc::new(Cell::new(false));
    let tree = replace_tree_random_internal(
        from,
        path,
        new_symbol,
        rng,
        None,
        None,
        Rc::clone(&error),
    );
    if error.get() {
        None
    } else {
        Some(tree)
    }
}

fn replace_tree_random_internal(
    from: &GpTree,
    path: &[usize],
    new_symbol: Rc<GpSymbol>,
    rng: &mut impl Rng,
    parent_depth: Option<i32>,
    parent_max_depth: Option<i32>,
    error: Rc<Cell<bool>>,
) -> GpTree {
    if path.is_empty() {
        return clone_tree(from, from.ref_(), parent_depth, parent_max_depth, error);
    }
    let (depth, max_depth) = calc_depth_max(parent_depth, parent_max_depth, &from.ref_());
    if path.len() == 1 {
        let replace_index = path[0];
        let mut symbols = Vec::new();
        let mut deps = Vec::new();
        for (idx, child) in from.children.iter().enumerate() {
            if idx == replace_index {
                symbols.push(new_symbol.clone());
                deps.push(new_symbol.min_depth());
            } else {
                symbols.push(child.ref_().symbol());
                deps.push(child.tree_depth());
            }
        }

        let parent_symbol = from.ref_().symbol();
        let choice = select_choice(&parent_symbol, from.choice.as_ref(), &symbols, &deps, depth, max_depth, rng);
        if let Some(choice) = choice {
            let mut node = build_node(
                from.ref_(),
                Some(choice.clone()),
                TreeType::Choice,
                parent_depth,
                parent_max_depth,
                error,
            );
            for (idx, ref_) in choice.refs().iter().enumerate() {
                if idx == replace_index {
                    let child = build_random_tree(Rc::clone(ref_), Some(node.depth), Some(node.max_depth), Rc::clone(&node.error), rng);
                    node.children.push(child);
                } else if let Some(old_child) = from.children.get(idx) {
                    let clone = clone_tree(old_child, Rc::clone(ref_), Some(node.depth), Some(node.max_depth), Rc::clone(&node.error));
                    node.children.push(clone);
                }
            }
            node
        } else {
            let node = build_node(from.ref_(), None, TreeType::Empty, parent_depth, parent_max_depth, error);
            node.error.set(true);
            node
        }
    } else {
        let mut node = build_node(
            from.ref_(),
            from.choice.clone(),
            from.tree_type,
            parent_depth,
            parent_max_depth,
            error,
        );
        for (idx, child) in from.children.iter().enumerate() {
            if idx == path[0] {
                let sub = replace_tree_random_internal(
                    child,
                    &path[1..],
                    new_symbol.clone(),
                    rng,
                    Some(node.depth),
                    Some(node.max_depth),
                    Rc::clone(&node.error),
                );
                node.children.push(sub);
            } else {
                let clone = clone_tree(
                    child,
                    child.ref_(),
                    Some(node.depth),
                    Some(node.max_depth),
                    Rc::clone(&node.error),
                );
                node.children.push(clone);
            }
        }
        node
    }
}

fn select_choice(
    symbol: &Rc<GpSymbol>,
    current: Option<&Rc<GpChoice>>,
    symbols: &[Rc<GpSymbol>],
    deps: &[i32],
    depth: i32,
    max_depth: i32,
    rng: &mut impl Rng,
) -> Option<Rc<GpChoice>> {
    let mut candidates = Vec::new();
    for choice in symbol.choices() {
        if choice.matches(symbols) {
            candidates.push(Rc::clone(choice));
        }
    }

    let mut candidate = current
        .and_then(|c| if c.matches(symbols) { Some(Rc::clone(c)) } else { None });

    while !candidates.is_empty() {
        if candidate.is_none() {
            let idx = rng.gen_range(0..candidates.len());
            candidate = Some(candidates[idx].clone());
        }
        let choice = candidate.take().unwrap();
        let mut ok = true;
        for (idx, reference) in choice.refs().iter().enumerate() {
            let maxd = if reference.param_max_depth() >= 0 {
                max_depth.min(depth + 1 + reference.param_max_depth())
            } else {
                max_depth
            };
            if depth + 1 + deps[idx] > maxd {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(choice);
        }
        candidates.retain(|c| !Rc::ptr_eq(c, &choice));
    }

    None
}

fn apply_indent(text: &str, indent: usize) -> String {
    if indent == 0 {
        return text.to_string();
    }
    let mut out = String::new();
    for ch in text.chars() {
        out.push(ch);
        if ch == '\r' {
            out.push_str(&" ".repeat(indent));
        }
    }
    out
}

fn current_line_indent(text: &str) -> usize {
    let mut count = 0usize;
    for ch in text.chars().rev() {
        if ch == '\r' {
            break;
        }
        count += 1;
    }
    count
}

fn node_name(node: &Node) -> String {
    String::from_utf8_lossy(node.name()).to_string()
}

fn node_str(node: &Node, name: &str) -> String {
    String::from_utf8_lossy(&node.get_str(name)).to_string()
}

fn node_int(node: &Node, name: &str) -> i32 {
    node.get_int(name)
}

fn node_real(node: &Node, name: &str) -> f64 {
    node.get_real(name)
}

fn node_bool(node: &Node, name: &str) -> bool {
    node.get_bool(name)
}

fn decimals_from_str(value: &str) -> usize {
    let value = value.trim();
    let dot = match value.find('.') {
        Some(idx) => idx,
        None => return 0,
    };
    let frac = &value[dot + 1..];
    let frac = frac.trim_end_matches('0');
    frac.len()
}

fn decimals_from_float(value: f64) -> usize {
    decimals_from_str(&value.to_string())
}

fn round_to_decimals(value: f64, decimals: usize) -> f64 {
    if decimals == 0 || !value.is_finite() {
        return value;
    }
    let factor = 10f64.powi(decimals as i32);
    (value * factor).round() / factor
}

fn format_float_raw(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    let exp = abs.log10().floor() as i32;
    let scale_exp = 15 - 1 - exp;
    let scale = 10f64.powi(scale_exp);
    let rounded = (value * scale).round() / scale;
    let mut s = rounded.to_string();
    if let Some(dot) = s.find('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.len() == dot + 1 {
            s.pop();
        }
    }
    s
}

fn format_float_with_decimals(value: f64, decimals: usize) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    let rounded = if decimals == 0 {
        value.round()
    } else {
        round_to_decimals(value, decimals)
    };
    let mut s = if decimals == 0 {
        format!("{:.0}", rounded)
    } else {
        format!("{:.*}", decimals, rounded)
    };
    if let Some(dot) = s.find('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.len() == dot + 1 {
            s.pop();
        }
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

fn node_set_str(node: &mut Node, name: &str, value: &str) {
    node.set_str(name, value.as_bytes().to_vec());
}

fn node_set_int(node: &mut Node, name: &str, value: i32) {
    node.set_int(name, value);
}

fn node_set_bool(node: &mut Node, name: &str, value: bool) {
    node.set_bool(name, value);
}

#[derive(Debug, Clone)]
struct NumberParams {
    name: String,
    from: f64,
    to: f64,
    inc: f64,
    optimize: bool,
}

fn is_ref_to_number(ref_cfg: &Node) -> Option<NumberParams> {
    let params_node = ref_cfg.child("PARAMS")?;
    let from_str = node_str(params_node, "from");
    let to_str = node_str(params_node, "to");
    let mut from = from_str.parse::<f64>().ok()?;
    let mut to = to_str.parse::<f64>().ok()?;
    if from > to {
        std::mem::swap(&mut from, &mut to);
    }
    let inc = node_str(params_node, "inc")
        .parse::<f64>()
        .unwrap_or(1.0)
        .abs();
    let optimize = node_str(params_node, "optimize").eq_ignore_ascii_case("true")
        || node_str(params_node, "opt").eq_ignore_ascii_case("true");
    let name = node_str(params_node, "name");
    if inc > 0.0 {
        Some(NumberParams {
            name,
            from,
            to,
            inc,
            optimize,
        })
    } else {
        None
    }
}

fn commented_text(text: &[u8], state: &mut i32, index: &mut usize) -> bool {
    if *index >= text.len() {
        return false;
    }
    let ch = text[*index];
    if *state == 0 {
        if ch == b'{' {
            *state = 1;
        } else if *index + 1 < text.len() {
            if ch == b'/' && text[*index + 1] == b'/' {
                *state = 2;
                *index += 1;
            } else if ch == b'(' && text[*index + 1] == b'*' {
                *state = 3;
                *index += 1;
            }
        }
    } else {
        match *state {
            1 => {
                if ch == b'}' {
                    *state = 0;
                }
            }
            2 => {
                if ch == b'\r' {
                    *state = 0;
                }
            }
            3 => {
                if *index + 1 < text.len() && ch == b'*' && text[*index + 1] == b')' {
                    *state = 0;
                    *index += 1;
                }
            }
            _ => {}
        }
    }

    if *state != 0 {
        *index += 1;
        return true;
    }
    false
}

pub fn parse_params(text: &str) -> (bool, Vec<String>, Vec<String>) {
    let bytes = text.as_bytes();
    let mut names = Vec::new();
    let mut values = Vec::new();
    let mut i: usize = 0;
    let mut name = false;
    let mut equal = false;
    let mut value = false;
    let mut quote = false;
    let mut dblquote = false;
    let mut n1: usize = 0;
    let mut n2: usize = 0;
    let mut v1: usize = 0;
    let mut v2: usize = 0;

    while i < bytes.len() {
        let ch = bytes[i] as char;
        if !name && ch != ' ' {
            name = true;
            equal = false;
            n1 = i;
            n2 = i;
        } else if name && !equal {
            if ch == '=' {
                equal = true;
                quote = false;
                dblquote = false;
                value = false;
            } else if ch != ' ' {
                n2 = i;
            }
        } else if name && equal && !value {
            if !quote && !dblquote {
                if ch == '\'' {
                    quote = true;
                } else if ch == '"' {
                    dblquote = true;
                } else if ch != ' ' {
                    value = true;
                    v1 = i;
                    v2 = i;
                }
            } else if (quote && ch == '\'') || (dblquote && ch == '"') {
                names.push(String::from_utf8_lossy(&bytes[n1..=n2]).to_string());
                values.push(String::new());
                name = false;
            } else if quote || dblquote || ch != ' ' {
                value = true;
                v1 = i;
                v2 = i;
            }
        } else if name && equal && value {
            if (quote && ch == '\'') || (dblquote && ch == '"') {
                names.push(String::from_utf8_lossy(&bytes[n1..=n2]).to_string());
                values.push(String::from_utf8_lossy(&bytes[v1..=v2]).to_string());
                name = false;
            } else if quote || dblquote || ch != ' ' {
                v2 = i;
            } else if ch == ' ' {
                names.push(String::from_utf8_lossy(&bytes[n1..=n2]).to_string());
                values.push(String::from_utf8_lossy(&bytes[v1..=v2]).to_string());
                name = false;
            }
        }
        i += 1;
    }

    if name && equal && value && !quote && !dblquote {
        names.push(String::from_utf8_lossy(&bytes[n1..=n2]).to_string());
        values.push(String::from_utf8_lossy(&bytes[v1..=v2]).to_string());
        name = false;
    }

    (!name, names, values)
}

pub fn parse_text(choice_cfg: &mut Node) {
    let mut code = node_str(choice_cfg, "Text");
    let mut refs_node = choice_cfg.get_or_create_child("REFS");
    refs_node.clear_all();

    let mut tag = false;
    let mut par = false;
    let mut chg = false;
    let mut cmt = 0;
    let mut tagcnt: i32 = 0;
    let mut t1: usize = 0;
    let mut p1: usize = 0;

    let mut i: usize = 0;
    while i < code.len() {
        let bytes = code.as_bytes();
        if commented_text(bytes, &mut cmt, &mut i) {
            continue;
        }
        if !tag && i + 1 < bytes.len() && bytes[i] == b'<' {
            if bytes[i + 1] != b'<' {
                tag = true;
                t1 = i + 1;
            } else {
                i += 1;
            }
        } else if tag {
            if !par && bytes[i] == b' ' {
                par = true;
                p1 = i;
            }
            if bytes[i] == b'>' {
                if i + 1 == bytes.len() || bytes[i + 1] != b'>' {
                    let (tagname, tagparams) = if !par {
                        (
                            code[t1..i].to_string(),
                            String::new(),
                        )
                    } else {
                        (
                            code[t1..p1].to_string(),
                            code[p1 + 1..i].to_string(),
                        )
                    };

                    let mut ref_node = Node::new(tagcnt.to_string());
                    node_set_int(&mut ref_node, "Pos", (t1 + 1) as i32);
                    node_set_int(&mut ref_node, "Len", (i - t1) as i32);
                    parse_cfg_params(&tagname, &tagparams, &mut ref_node);

                    let mut final_tagname = tagname.clone();
                    if let Some(params) = is_ref_to_number(&ref_node) {
                        if params.optimize {
                            final_tagname = format!(
                                "{}_{}_{}_{}_opt",
                                params.name, params.from, params.to, params.inc
                            );
                        } else {
                            final_tagname =
                                format!("{}_{}_{}_{}", params.name, params.from, params.to, params.inc);
                        }
                    }

                    if final_tagname.eq_ignore_ascii_case("START") {
                        final_tagname.push('_');
                        if !par {
                            code.insert(i, '_');
                        } else {
                            code.insert(p1, '_');
                        }
                        i += 1;
                        chg = true;
                    }

                    let group = refs_node.get_or_create_child(final_tagname.as_str());
                    group.set_sorted(false);
                    let mut child = Node::new(tagcnt.to_string());
                    child.copy_from(&ref_node, true, true);
                    group.add_child(child);
                    tagcnt += 1;

                    par = false;
                    tag = false;
                } else if i + 1 < bytes.len() {
                    i += 1;
                }
            } else if bytes[i] == b'\r' {
                tag = false;
                par = false;
            }
        }
        i += 1;
    }

    if chg {
        node_set_str(choice_cfg, "Text", &code);
    }
}

fn parse_cfg_params(name: &str, text: &str, ref_cfg: &mut Node) {
    if let Some(params) = ref_cfg.child_mut("PARAMS") {
        params.clear_all();
    }
    let (ok, names, values) = parse_params(text);
    if !ok || names.is_empty() {
        return;
    }
    let params_node = ref_cfg.get_or_create_child("PARAMS");
    node_set_str(params_node, "name", name);
    for (n, v) in names.iter().zip(values.iter()) {
        node_set_str(params_node, n, v);
    }
}

pub fn calc_lengths(grammar_cfg: &mut Node) -> Result<(), GggpError> {
    let max_depth = node_int(grammar_cfg, "MaxDepth");
    let rules = grammar_cfg.child_mut("RULES").ok_or(GggpError::MissingRules)?;
    if rules.children().is_empty() {
        return Err(GggpError::InvalidConfig("empty grammar"));
    }

    let mut virtual_symbols: Vec<String> = Vec::new();

    for symbol in rules.children_mut() {
        node_set_int(symbol, "Length", 0);
        let choices = symbol
            .child_mut("CHOICES")
            .ok_or_else(|| GggpError::InvalidConfig("symbol without choices"))?;
        if choices.children().is_empty() {
            return Err(GggpError::InvalidConfig("symbol has no rules"));
        }

        for choice in choices.children_mut() {
            if choice.child("REFS").map(|r| r.children().is_empty()).unwrap_or(true) {
                node_set_int(choice, "Length", 1);
            } else {
                node_set_int(choice, "Length", 0);
                if let Some(refs) = choice.child_mut("REFS") {
                    for group in refs.children_mut() {
                        node_set_int(group, "Length", 0);
                        let mut is_virtual = false;
                        for reference in group.children() {
                            if is_ref_to_number(reference).is_some() {
                                is_virtual = true;
                                break;
                            }
                        }
                        if is_virtual {
                            virtual_symbols.push(node_name(group));
                        }
                    }
                }
            }
        }
    }

    for sym in &virtual_symbols {
        set_symbol_length(rules, sym, 2);
    }

    let mut depth = 1;
    let mut remaining = rules.children().len();
    while remaining > 0 {
        let mut remaining_next = remaining;
        let symbol_count = rules.child_count();
        for idx in 0..symbol_count {
            let mut update_symbol: Option<String> = None;
            {
                let symbol = match rules.child_by_index_mut(idx) {
                    Some(sym) => sym,
                    None => continue,
                };
                if node_int(symbol, "Length") == 0 {
                    if let Some(choices) = symbol.child_mut("CHOICES") {
                        for choice in choices.children_mut() {
                            if node_int(choice, "Length") == depth {
                                node_set_int(symbol, "Length", depth);
                                update_symbol = Some(node_name(symbol));
                                remaining_next -= 1;
                                break;
                            }
                        }
                    }
                }
            }
            if let Some(sym_name) = update_symbol {
                set_symbol_length(rules, &sym_name, depth + 1);
            }
        }
        remaining = remaining_next;
        if remaining == 0 {
            break;
        }
        if depth >= max_depth {
            return Err(GggpError::InvalidConfig(
                "config contains infinite circular references",
            ));
        }
        depth += 1;
    }

    Ok(())
}

fn set_symbol_length(rules: &mut Node, symbol_name: &str, symbol_length: i32) {
    for symbol in rules.children_mut() {
        if let Some(choices) = symbol.child_mut("CHOICES") {
            for choice in choices.children_mut() {
                if node_int(choice, "Length") == 0 {
                    let mut ok = true;
                    let mut max_len = 0;
                    if let Some(refs) = choice.child_mut("REFS") {
                        for group in refs.children_mut() {
                            if group.name().eq_ignore_ascii_case(symbol_name.as_bytes()) {
                                node_set_int(group, "Length", symbol_length);
                                max_len = max_len.max(symbol_length);
                            } else {
                                let length = node_int(group, "Length");
                                if length == 0 {
                                    ok = false;
                                } else {
                                    max_len = max_len.max(length);
                                }
                            }
                        }
                    }
                    if ok {
                        node_set_int(choice, "Length", max_len);
                    }
                }
            }
        }
    }
}

pub fn str_hash_key(data: &[u8]) -> u32 {
    let mut hash: u32 = 0;
    for &b in data.iter().rev() {
        hash = hash.wrapping_shl(5).wrapping_add(hash).wrapping_add(b as u32);
    }
    hash
}

#[derive(Debug, Clone)]
pub struct GpData {
    pub groups_count: usize,
    pub groups_names: Vec<String>,
    pub groups_inputs_list: Vec<Vec<i32>>,
    pub inputs_names: Vec<String>,
    pub inputs_used: Vec<bool>,
    pub inputs_data_count: usize,
    pub inputs_objects: Vec<Option<usize>>,
    pub inputs_values: Vec<Option<usize>>,
    pub inputs_current_index: Vec<i32>,
    pub inputs_base_offset: Vec<i32>,
    pub inputs_data_offset: Vec<i32>,
    pub inputs_allow_calc: Vec<bool>,
    pub inputs_bar_complete: Vec<bool>,
}

pub fn text_replace_groups_tags(text: &str, data: &mut GpData) -> Result<String, GggpError> {
    let mut result = String::new();
    let mut text_lc = text.to_lowercase();
    let mut sp = 0usize;
    let mut cmt = 0;

    while sp + 8 <= text.len() {
        if commented_text(text_lc.as_bytes(), &mut cmt, &mut sp) {
            continue;
        }
        let x1 = find_substring(&text_lc, "%group", sp);
        if x1.is_none() {
            break;
        }
        let x1 = x1.unwrap();
        let x2 = find_substring(&text_lc, "%", x1 + 1);
        if x2.is_none() {
            break;
        }
        let x2 = x2.unwrap();
        let (ok, names, values) = parse_params(&text[x1 + 1..x2]);
        if ok && names.len() == 2 && names[0].eq_ignore_ascii_case("group") && names[1].eq_ignore_ascii_case("index") {
            let idx_str = values[1].split('{').next().unwrap_or(&values[1]);
            if let Ok(il) = idx_str.parse::<usize>() {
                let group_index = group_index(&values[0], data)?;
                if let Some(group_list) = data.groups_inputs_list.get(group_index) {
                    if il < group_list.len() {
                        let ii = group_list[il] as usize;
                        result.push_str(&text[sp..x1]);
                        result.push_str(&format!("{}{{{}}}", ii, data.inputs_names[ii]));
                        data.inputs_used[ii] = true;
                    } else {
                        result.push_str(&text[sp..=x2]);
                    }
                } else {
                    result.push_str(&text[sp..=x2]);
                }
            } else {
                result.push_str(&text[sp..=x2]);
            }
        } else {
            result.push_str(&text[sp..=x2]);
        }
        sp = x2 + 1;
    }
    result.push_str(&text[sp..]);

    let mut final_result = String::new();
    let mut text_lc = result.to_lowercase();
    let mut sp = 0usize;
    cmt = 0;
    while sp + 6 <= result.len() {
        if commented_text(text_lc.as_bytes(), &mut cmt, &mut sp) {
            continue;
        }
        let x1 = find_substring(&text_lc, "%ind", sp);
        if x1.is_none() {
            break;
        }
        let x1 = x1.unwrap();
        let x2 = find_substring(&text_lc, "%", x1 + 1);
        if x2.is_none() {
            break;
        }
        let x2 = x2.unwrap();
        let (ok, names, values) = parse_params(&result[x1 + 1..x2]);
        if ok && names.len() == 2 && names[0].eq_ignore_ascii_case("ind") && names[1].eq_ignore_ascii_case("index") {
            let idx_str = values[1].split('{').next().unwrap_or(&values[1]);
            if let Ok(il) = idx_str.parse::<usize>() {
                if il < data.inputs_names.len() {
                    final_result.push_str(&result[sp..x1]);
                    final_result.push_str(&format!("{}{{{}}}", il, values[0]));
                    data.inputs_used[il] = true;
                } else {
                    final_result.push_str(&result[sp..=x2]);
                }
            } else {
                final_result.push_str(&result[sp..=x2]);
            }
        } else {
            final_result.push_str(&result[sp..=x2]);
        }
        sp = x2 + 1;
    }
    final_result.push_str(&result[sp..]);

    Ok(final_result)
}

pub fn gp_data_from_config(
    project_cfg: &Node,
    global_cfg: Option<&Node>,
    include_inactive: bool,
) -> Result<GpData, GggpError> {
    let cache = build_indicator_cache(project_cfg, include_inactive)?;
    let groups_node = project_cfg
        .child("GRAMMAR_GROUPS")
        .ok_or(GggpError::InvalidConfig("project config missing GRAMMAR_GROUPS"))?;
    let global_cfg = global_cfg.unwrap_or(project_cfg);

    let mut groups_names = Vec::new();
    let mut groups_inputs_list = Vec::new();
    for group in groups_node.children() {
        if !node_bool(group, "Active") {
            continue;
        }
        let name = node_str(group, "Name");
        let path = node_str(group, "Path");
        if name.is_empty() || path.is_empty() {
            continue;
        }
        let list = group_inputs_list_for_path(&path, project_cfg, global_cfg, &cache)?;
        groups_names.push(name);
        groups_inputs_list.push(list);
    }

    let inputs_names = if let Some(inputs_names) = cache.inputs_names.as_ref() {
        inputs_names.clone()
    } else {
        let mut names = Vec::new();
        for name in &cache.dat_names {
            names.push(name.clone());
        }
        for entry in &cache.idx_list {
            let tail = entry.full_path.clone();
            if tail.starts_with('>') {
                names.push(format!("DATA{}{}", entry.data_index, tail));
            } else {
                names.push(format!("DATA{}>{}", entry.data_index, tail));
            }
        }
        names
    };

    let inputs_len = inputs_names.len();
    Ok(GpData {
        groups_count: groups_names.len(),
        groups_names,
        groups_inputs_list,
        inputs_names,
        inputs_used: vec![false; inputs_len],
        inputs_data_count: inputs_len,
        inputs_objects: vec![None; inputs_len],
        inputs_values: vec![None; inputs_len],
        inputs_current_index: vec![0; inputs_len],
        inputs_base_offset: vec![0; inputs_len],
        inputs_data_offset: vec![0; inputs_len],
        inputs_allow_calc: vec![false; inputs_len],
        inputs_bar_complete: vec![false; inputs_len],
    })
}

#[derive(Debug, Clone)]
pub struct IndicatorEntry {
    pub full_path: String,
    pub data_index: i32,
    pub inp_index: usize,
    pub input_index: Option<usize>,
    pub paramless_path: String,
}

struct IndicatorCache {
    dat_names: Vec<String>,
    inp_list: Vec<String>,
    idx_list: Vec<IndicatorEntry>,
    inputs_names: Option<Vec<String>>,
    input_name_to_index: Option<HashMap<String, usize>>,
}

#[derive(Debug)]
pub struct ParsedGroupPath<'a> {
    pub conf: &'a Node,
    pub collect_type: i32,
    pub allowed_data: Vec<i32>,
}

pub fn grammar_config_replace_group_tags(
    project_cfg: &Node,
    grammar_cfg: &mut Node,
    global_cfg: Option<&Node>,
    cleanup: bool,
) -> Result<(), GggpError> {
    grammar_config_replace_group_tags_with_options(
        project_cfg,
        grammar_cfg,
        global_cfg,
        cleanup,
        false,
    )
}

pub fn grammar_config_replace_group_tags_with_options(
    project_cfg: &Node,
    grammar_cfg: &mut Node,
    global_cfg: Option<&Node>,
    cleanup: bool,
    include_inactive: bool,
) -> Result<(), GggpError> {
    let rules = grammar_cfg
        .child_mut("RULES")
        .ok_or(GggpError::MissingRules)?;
    let mut group_cache: HashMap<String, usize> = HashMap::new();
    let mut indicator_cache: Option<IndicatorCache> = None;
    let global_cfg = global_cfg.unwrap_or(project_cfg);

    let symbol_count = rules.child_count();
    for si in 0..symbol_count {
        let choices_count = {
            let symbol = match rules.child_by_index_mut(si) {
                Some(sym) => sym,
                None => continue,
            };
            match symbol.child_mut("CHOICES") {
                Some(choices) => choices.child_count(),
                None => continue,
            }
        };

        for ci in 0..choices_count {
            let mut choice_text = String::new();
            {
                let symbol = match rules.child_by_index_mut(si) {
                    Some(sym) => sym,
                    None => continue,
                };
                let choices = match symbol.child_mut("CHOICES") {
                    Some(choices) => choices,
                    None => continue,
                };
                let choice = match choices.child_by_index_mut(ci) {
                    Some(choice) => choice,
                    None => continue,
                };
                choice_text = node_str(choice, "Text");
            }

            let mut any_changed = false;
            let mut txt = choice_text;
            let mut txtlc = txt.to_lowercase();

            let mut txtn = String::new();
            let mut sp = 0usize;
            let mut cmt = 0;
            let mut changed = false;
            while sp + 8 <= txt.len() {
                if commented_text(txtlc.as_bytes(), &mut cmt, &mut sp) {
                    continue;
                }
                let x1 = match find_substring(&txtlc, "%group", sp) {
                    Some(pos) => pos,
                    None => break,
                };
                let x2 = match find_substring(&txtlc, "%", x1 + 1) {
                    Some(pos) => pos,
                    None => break,
                };
                let (_, names, values) = parse_params(&txt[x1 + 1..x2]);
                if !names.is_empty() && names[0].eq_ignore_ascii_case("group") {
                    let group_name = values.get(0).cloned().unwrap_or_default();
                    let replacement = if cleanup {
                        format!("%group=\"{}\"%", group_name)
                    } else {
                        let cache = ensure_indicator_cache(
                            &mut indicator_cache,
                            project_cfg,
                            include_inactive,
                        )?;
                        let total = number_of_group_series(
                            &group_name,
                            project_cfg,
                            global_cfg,
                            cache,
                            &mut group_cache,
                        )?;
                        format!(
                            "%group=\"{}\" index=<int from=0 to={}>%",
                            group_name,
                            total.saturating_sub(1)
                        )
                    };
                    txtn.push_str(&txt[sp..x1]);
                    txtn.push_str(&replacement);
                    changed = true;
                } else {
                    txtn.push_str(&txt[sp..=x2]);
                }
                sp = x2 + 1;
            }
            if changed {
                any_changed = true;
                txt = format!("{}{}", txtn, &txt[sp..]);
                txtlc = txt.to_lowercase();
                txtn = String::new();
            }

            sp = 0;
            cmt = 0;
            changed = false;
            while sp + 6 <= txt.len() {
                if commented_text(txtlc.as_bytes(), &mut cmt, &mut sp) {
                    continue;
                }
                let x1 = match find_substring(&txtlc, "%ind", sp) {
                    Some(pos) => pos,
                    None => break,
                };
                let x2 = match find_substring(&txtlc, "%", x1 + 1) {
                    Some(pos) => pos,
                    None => break,
                };
                let (_, names, values) = parse_params(&txt[x1 + 1..x2]);
                if !names.is_empty() && names[0].eq_ignore_ascii_case("ind") {
                    let ind_name = values.get(0).cloned().unwrap_or_default();
                    let replacement = if cleanup {
                        format!("%ind=\"{}\"%", ind_name)
                    } else {
                        let cache = ensure_indicator_cache(
                            &mut indicator_cache,
                            project_cfg,
                            include_inactive,
                        )?;
                        let idx = index_of_indicator(&ind_name, cache)?;
                        format!("%ind=\"{}\" index={}%", ind_name, idx)
                    };
                    txtn.push_str(&txt[sp..x1]);
                    txtn.push_str(&replacement);
                    changed = true;
                } else {
                    txtn.push_str(&txt[sp..=x2]);
                }
                sp = x2 + 1;
            }

            let final_text = if changed {
                any_changed = true;
                format!("{}{}", txtn, &txt[sp..])
            } else if any_changed {
                txt
            } else {
                txt
            };

            if any_changed {
                let symbol = match rules.child_by_index_mut(si) {
                    Some(sym) => sym,
                    None => continue,
                };
                let choices = match symbol.child_mut("CHOICES") {
                    Some(choices) => choices,
                    None => continue,
                };
                let choice = match choices.child_by_index_mut(ci) {
                    Some(choice) => choice,
                    None => continue,
                };
                node_set_str(choice, "Text", &final_text);
                parse_text(choice);
            }
        }
    }

    calc_lengths(grammar_cfg)?;
    Ok(())
}

pub fn parse_group_path<'a>(
    path: &str,
    global_cfg: &'a Node,
    local_cfg: &'a Node,
) -> Option<ParsedGroupPath<'a>> {
    if path.trim().is_empty() {
        return None;
    }

    let mut allowed_data = Vec::new();
    let mut rest = path;
    if let Some(pos) = path.find('>') {
        let prefix = path[..pos].trim();
        for token in prefix.split(',') {
            let token = token.trim();
            if let Some(name) = token.get(0..4) {
                if name.eq_ignore_ascii_case("DATA") {
                    if let Ok(value) = token[4..].trim().parse::<i32>() {
                        allowed_data.push(value);
                    }
                }
            }
        }
        if allowed_data.is_empty() {
            allowed_data.clear();
        }
        rest = &path[pos + 1..];
    }

    let mut conf = if rest.starts_with('/') {
        rest = &rest[1..];
        global_cfg
    } else {
        local_cfg
    };

    let mut collect_type = 0;
    for part in rest.split('/') {
        let mut part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part == "*" {
            collect_type = 1;
            break;
        }
        if part.ends_with('*') {
            collect_type = 2;
            part = part.trim_end_matches('*').trim();
        }
        if part.is_empty() {
            continue;
        }
        let groups = conf.child("GROUPS")?;
        conf = groups.child(part)?;
    }

    Some(ParsedGroupPath {
        conf,
        collect_type,
        allowed_data,
    })
}

pub fn group_indicators_recursive(list: &mut Vec<String>, conf: &Node, collect_type: i32) {
    let mut seen: HashSet<String> = list.iter().map(|s| s.to_lowercase()).collect();
    group_indicators_recursive_internal(list, &mut seen, conf, collect_type);
}

fn group_indicators_recursive_internal(
    list: &mut Vec<String>,
    seen: &mut HashSet<String>,
    conf: &Node,
    collect_type: i32,
) {
    if conf.name().is_empty() {
        return;
    }
    if (collect_type == 0 || collect_type == 2) {
        if let Some(indicators) = conf.child("INDICATORS") {
            for indicator in indicators.children() {
                group_indicators_indicator_recursive(list, seen, indicator, "");
            }
        }
    }
    if (collect_type == 1 || collect_type == 2) {
        if let Some(groups) = conf.child("GROUPS") {
            for group in groups.children() {
                group_indicators_recursive_internal(list, seen, group, 2);
            }
        }
    }
}

fn group_indicators_indicator_recursive(
    list: &mut Vec<String>,
    seen: &mut HashSet<String>,
    conf: &Node,
    path: &str,
) {
    let name = node_str(conf, "Name");
    let path = if path.is_empty() {
        name
    } else {
        format!("{}>{}", path, name)
    };

    let mut plot_count = 0usize;
    if let Some(plots) = conf.child("PLOTS") {
        plot_count = plots.child_count();
        for plot in plots.children() {
            let num = node_str(plot, "Num");
            let mut s = String::new();
            if plot_count == 1 {
                if conf.child("INDICATORS").is_none() && plot.child("INDICATORS").is_none() {
                    s = path.clone();
                }
            } else if node_bool(plot, "Group") && plot.child("INDICATORS").is_none() {
                s = format!("{}.p{}", path, num);
            }
            if !s.is_empty() {
                let key = s.to_lowercase();
                if seen.insert(key) {
                    list.push(s);
                }
            }
            if let Some(plot_inds) = plot.child("INDICATORS") {
                let plot_path = format!("{}.p{}", path, num);
                for indicator in plot_inds.children() {
                    group_indicators_indicator_recursive(list, seen, indicator, &plot_path);
                }
            }
        }
    }

    if let Some(indicators) = conf.child("INDICATORS") {
        if plot_count < 2 {
            for indicator in indicators.children() {
                group_indicators_indicator_recursive(list, seen, indicator, &path);
            }
        } else {
            let base = format!("{}.p1", path);
            for indicator in indicators.children() {
                group_indicators_indicator_recursive(list, seen, indicator, &base);
            }
        }
    }
}

pub fn input_indicators_recursive(
    list: &mut Vec<String>,
    index_list: &mut Vec<IndicatorEntry>,
    conf: &Node,
    is_indicator: bool,
    data_index: i32,
    with_params: bool,
    dataless: bool,
    paramless_list: Option<&[String]>,
    path: &str,
    paramless_path: &str,
    dataless_path: &str,
    with_plot_nums: bool,
    require_unique: bool,
) {
    input_indicators_recursive_internal(
        list,
        index_list,
        conf,
        is_indicator,
        data_index,
        with_params,
        dataless,
        paramless_list,
        path,
        paramless_path,
        dataless_path,
        with_plot_nums,
        require_unique,
        false,
    );
}

fn input_indicators_recursive_internal(
    list: &mut Vec<String>,
    index_list: &mut Vec<IndicatorEntry>,
    conf: &Node,
    is_indicator: bool,
    data_index: i32,
    with_params: bool,
    dataless: bool,
    paramless_list: Option<&[String]>,
    path: &str,
    paramless_path: &str,
    dataless_path: &str,
    with_plot_nums: bool,
    require_unique: bool,
    include_inactive: bool,
) {
    if !is_indicator {
        if let Some(indicators) = conf.child("INDICATORS") {
            for indicator in indicators.children() {
                input_indicators_recursive_internal(
                    list,
                    index_list,
                    indicator,
                    true,
                    data_index,
                    with_params,
                    dataless,
                    paramless_list,
                    path,
                    paramless_path,
                    dataless_path,
                    with_plot_nums,
                    require_unique,
                    include_inactive,
                );
            }
        }
        return;
    }

    let params = build_indicator_params(conf, with_params, paramless_list);
    if params.is_empty() {
        input_indicators_emit(
            list,
            index_list,
            conf,
            data_index,
            with_params,
            dataless,
            paramless_list,
            path,
            paramless_path,
            dataless_path,
            &[],
            require_unique,
            include_inactive,
        );
    } else {
        let mut current: Vec<String> = Vec::with_capacity(params.len());
        let mut emit = |values: &[String]| {
            input_indicators_emit(
                list,
                index_list,
                conf,
                data_index,
                with_params,
                dataless,
                paramless_list,
                path,
                paramless_path,
                dataless_path,
                values,
                require_unique,
                include_inactive,
            );
        };
        params_recursive(&params, 0, &mut current, &mut emit);
    }
}

fn input_indicators_emit(
    list: &mut Vec<String>,
    index_list: &mut Vec<IndicatorEntry>,
    conf: &Node,
    data_index: i32,
    with_params: bool,
    dataless: bool,
    paramless_list: Option<&[String]>,
    path: &str,
    paramless_path: &str,
    dataless_path: &str,
    params_comb: &[String],
    require_unique: bool,
    include_inactive: bool,
) {
    let name = node_str(conf, "Name");
    let pa = if paramless_path.is_empty() {
        name.clone()
    } else {
        format!("{}>{}", paramless_path, name)
    };

    let mut params_text = String::new();
    for (idx, item) in params_comb.iter().enumerate() {
        if idx > 0 {
            params_text.push(',');
        }
        params_text.push_str(item);
    }
    let mut pp = if params_text.is_empty() {
        name.clone()
    } else {
        format!("{}({})", name, params_text)
    };

    let base = if with_params { pp.clone() } else { name.clone() };
    let d = if dataless_path.is_empty() {
        base
    } else {
        format!("{}>{}", dataless_path, base)
    };
    if !path.is_empty() {
        pp = format!("{}>{}", path, pp);
    }

    let mut plot_count = 0usize;
    if let Some(plots) = conf.child("PLOTS") {
        plot_count = plots.child_count();
        for plot in plots.children() {
            let num = node_str(plot, "Num");
            if node_bool(plot, "Active") || include_inactive {
                let (da, pa2, ppa) = if plot_count == 1 {
                    (d.clone(), pa.clone(), pp.clone())
                } else {
                    (
                        format!("{}.p{}", d, num),
                        format!("{}.p{}", pa, num),
                        format!("{}.p{}", pp, num),
                    )
                };

                let list_value = if with_params {
                    if dataless { da.clone() } else { ppa.clone() }
                } else if dataless {
                    da.clone()
                } else {
                    pa2.clone()
                };

            add_indicator_entry(
                list,
                index_list,
                list_value,
                &ppa,
                data_index,
                require_unique,
                &pa2,
            );
            }

            if let Some(plot_inds) = plot.child("INDICATORS") {
                let ppa_path = format!("{}.p{}", pp, num);
                let pa_path = format!("{}.p{}", pa, num);
                let da_path = format!("{}.p{}", d, num);
                for indicator in plot_inds.children() {
                    input_indicators_recursive_internal(
                        list,
                        index_list,
                        indicator,
                        true,
                        data_index,
                        with_params,
                        dataless,
                        paramless_list,
                        &ppa_path,
                        &pa_path,
                        &da_path,
                        false,
                        require_unique,
                        include_inactive,
                    );
                }
            }
        }
    }

    if let Some(indicators) = conf.child("INDICATORS") {
        if plot_count < 2 {
            for indicator in indicators.children() {
                input_indicators_recursive_internal(
                    list,
                    index_list,
                    indicator,
                    true,
                    data_index,
                    with_params,
                    dataless,
                    paramless_list,
                    &pp,
                    &pa,
                    &d,
                    false,
                    require_unique,
                    include_inactive,
                );
            }
        } else {
            let ppa_path = format!("{}.p1", pp);
            let pa_path = format!("{}.p1", pa);
            let da_path = format!("{}.p1", d);
            for indicator in indicators.children() {
                input_indicators_recursive_internal(
                    list,
                    index_list,
                    indicator,
                    true,
                    data_index,
                    with_params,
                    dataless,
                    paramless_list,
                    &ppa_path,
                    &pa_path,
                    &da_path,
                    false,
                    require_unique,
                    include_inactive,
                );
            }
        }
    }
}

fn add_indicator_entry(
    list: &mut Vec<String>,
    index_list: &mut Vec<IndicatorEntry>,
    list_value: String,
    full_path: &str,
    data_index: i32,
    require_unique: bool,
    paramless_path: &str,
) {
    if require_unique {
        if index_list.iter().any(|entry| {
            entry.data_index == data_index
                && entry.full_path.eq_ignore_ascii_case(full_path)
        }) {
            return;
        }
    }
    let inp_index = list.len();
    list.push(list_value);
    index_list.push(IndicatorEntry {
        full_path: full_path.to_string(),
        data_index,
        inp_index,
        input_index: None,
        paramless_path: paramless_path.to_string(),
    });
}

fn build_indicator_params(
    conf: &Node,
    with_params: bool,
    paramless_list: Option<&[String]>,
) -> Vec<Vec<String>> {
    let params_node = match conf.child("PARAMS") {
        Some(node) => node,
        None => return Vec::new(),
    };
    if params_node.children().is_empty() {
        return Vec::new();
    }
    let summarize = with_params && paramless_list.is_some();
    let mut params = Vec::new();
    for param in params_node.children() {
        let optimize = node_bool(param, "Optimize");
        let range = node_bool(param, "Range");
        if summarize {
            if !optimize {
                params.push(vec![node_str(param, "Value")]);
            } else if range {
                let start = node_str(param, "Range.Start");
                let end = node_str(param, "Range.End");
                let inc = node_str(param, "Range.Increment");
                let mut s = format!("{}-{}", start, end);
                s.push('`');
                s.push_str(&inc);
                params.push(vec![s]);
            } else {
                params.push(vec![node_str(param, "List")]);
            }
        } else if !optimize {
            params.push(vec![node_str(param, "Value")]);
        } else if range {
            let start_str = node_str(param, "Range.Start");
            let end_str = node_str(param, "Range.End");
            let inc_str = node_str(param, "Range.Increment");
            let mut start = start_str.parse::<f64>().unwrap_or(0.0);
            let end = end_str.parse::<f64>().unwrap_or(0.0);
            let inc = inc_str.parse::<f64>().unwrap_or(0.0).abs();
            let decimals = decimals_from_str(&start_str)
                .max(decimals_from_str(&end_str))
                .max(decimals_from_str(&inc_str));
            if inc <= 0.0 {
                params.push(vec![format_float_with_decimals(start, decimals)]);
            } else {
                let count = ((end - start).abs() / inc).ceil() as usize + 1;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(format_float_with_decimals(start, decimals));
                    start = round_to_decimals(start + inc, decimals);
                }
                params.push(values);
            }
        } else {
            let list = node_str(param, "List");
            let values = list
                .split('|')
                .map(|item| item.to_string())
                .collect::<Vec<_>>();
            params.push(values);
        }
    }
    params
}

fn params_recursive(
    params: &[Vec<String>],
    index: usize,
    current: &mut Vec<String>,
    emit: &mut dyn FnMut(&[String]),
) {
    if index >= params.len() {
        emit(current);
        return;
    }
    for value in &params[index] {
        if current.len() <= index {
            current.push(value.clone());
        } else {
            current[index] = value.clone();
        }
        params_recursive(params, index + 1, current, emit);
    }
}

fn ensure_indicator_cache<'a>(
    cache: &'a mut Option<IndicatorCache>,
    project_cfg: &Node,
    include_inactive: bool,
) -> Result<&'a IndicatorCache, GggpError> {
    if cache.is_none() {
        *cache = Some(build_indicator_cache(project_cfg, include_inactive)?);
    }
    Ok(cache.as_ref().expect("indicator cache"))
}

struct GrammarInputEntry {
    input_index: usize,
    full_name: String,
}

fn collect_grammar_inputs(
    node: &Node,
    data_index: i32,
    entries: &mut Vec<GrammarInputEntry>,
    seen: &mut HashSet<usize>,
) {
    if node.get_value("InputsIndex").is_some() {
        let index = node.get_int("InputsIndex");
        if index >= 0 {
            let input_index = index as usize;
            let mut full_name = node_str(node, "InputsNames");
            if full_name.is_empty() {
                let paramless_path = node_str(node, "InputsPaths");
                if !paramless_path.is_empty() {
                    full_name = format!("DATA{}>{}", data_index, paramless_path);
                }
            }
            if !full_name.is_empty() && seen.insert(input_index) {
                entries.push(GrammarInputEntry {
                    input_index,
                    full_name,
                });
            }
        }
    }
    for child in node.children() {
        collect_grammar_inputs(child, data_index, entries, seen);
    }
}

fn grammar_inputs_entries(project_cfg: &Node, include_inactive: bool) -> Vec<GrammarInputEntry> {
    let series = match project_cfg.child("SERIES") {
        Some(series) => series,
        None => return Vec::new(),
    };
    let mut entries: Vec<GrammarInputEntry> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();

    for serie in series.children() {
        if !include_inactive && !node_bool(serie, "Active") {
            continue;
        }
        let name = node_name(serie);
        let data_index = parse_data_index(&name);
        if let Some(grammar) = serie.child("Grammar") {
            collect_grammar_inputs(grammar, data_index, &mut entries, &mut seen);
        }
    }

    entries
}

fn build_indicator_cache(
    project_cfg: &Node,
    include_inactive: bool,
) -> Result<IndicatorCache, GggpError> {
    let series = project_cfg
        .child("SERIES")
        .ok_or(GggpError::InvalidConfig("project config missing SERIES"))?;
    let mut dat_names = Vec::new();
    let mut inp_list = Vec::new();
    let mut idx_list = Vec::new();
    let grammar_entries = grammar_inputs_entries(project_cfg, include_inactive);
    let mut grammar_index_by_name: HashMap<String, usize> = HashMap::new();

    for serie in series.children() {
        if !include_inactive && !node_bool(serie, "Active") {
            continue;
        }
        let name = node_name(serie);
        let data_index = parse_data_index(&name);
        dat_names.push(name);
        input_indicators_recursive_internal(
            &mut inp_list,
            &mut idx_list,
            serie,
            false,
            data_index,
            false,
            true,
            None,
            "",
            "",
            "",
            true,
            true,
            include_inactive,
        );
    }

    idx_list.sort_by(|a, b| {
        let a_key = a.full_path.to_lowercase();
        let b_key = b.full_path.to_lowercase();
        let cmp = a_key.cmp(&b_key);
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
        let a_bytes = a.data_index.to_le_bytes();
        let b_bytes = b.data_index.to_le_bytes();
        a_bytes.cmp(&b_bytes)
    });

    if !grammar_entries.is_empty() {
        for entry in &grammar_entries {
            grammar_index_by_name.insert(entry.full_name.to_lowercase(), entry.input_index);
        }
    }

    for entry in &mut idx_list {
        let key = full_indicator_name(entry).to_lowercase();
        if let Some(index) = grammar_index_by_name.get(&key).copied() {
            entry.input_index = Some(index);
        }
    }

    let mut next_index = dat_names.len();
    let mut max_assigned = dat_names.len().saturating_sub(1);
    let mut consistent = true;
    for entry in &mut idx_list {
        if let Some(index) = entry.input_index {
            if index < next_index {
                consistent = false;
                break;
            }
            max_assigned = max_assigned.max(index);
            next_index = index + 1;
        } else {
            entry.input_index = Some(next_index);
            max_assigned = max_assigned.max(next_index);
            next_index += 1;
        }
    }
    if !consistent {
        next_index = dat_names.len();
        max_assigned = dat_names.len().saturating_sub(1);
        for entry in &mut idx_list {
            entry.input_index = Some(next_index);
            max_assigned = max_assigned.max(next_index);
            next_index += 1;
        }
    }

    let mut names = vec![String::new(); max_assigned + 1];
    for (idx, name) in dat_names.iter().enumerate() {
        if idx < names.len() {
            names[idx] = name.clone();
        }
    }
    for entry in &idx_list {
        if let Some(index) = entry.input_index {
            if index >= names.len() {
                names.resize(index + 1, String::new());
            }
            if names[index].is_empty() {
                names[index] = full_indicator_name(entry);
            }
        }
    }

    let mut map = HashMap::new();
    for (idx, name) in names.iter().enumerate() {
        if !name.is_empty() {
            map.insert(name.to_lowercase(), idx);
        }
    }
    let inputs_names = Some(names);
    let input_name_to_index = Some(map);

    Ok(IndicatorCache {
        dat_names,
        inp_list,
        idx_list,
        inputs_names,
        input_name_to_index,
    })
}

fn number_of_group_series(
    group_name: &str,
    project_cfg: &Node,
    global_cfg: &Node,
    cache: &IndicatorCache,
    group_cache: &mut HashMap<String, usize>,
) -> Result<usize, GggpError> {
    let key = group_name.to_lowercase();
    if let Some(value) = group_cache.get(&key) {
        return Ok(*value);
    }

    let groups = project_cfg
        .child("GRAMMAR_GROUPS")
        .ok_or(GggpError::InvalidConfig("project config missing GRAMMAR_GROUPS"))?;
    let mut path = String::new();
    for group in groups.children() {
        if node_bool(group, "Active") && node_str(group, "Name").eq_ignore_ascii_case(group_name) {
            path = node_str(group, "Path");
            break;
        }
    }
    if path.is_empty() {
        return Err(GggpError::InvalidData(format!(
            "group '{}' is not found",
            group_name
        )));
    }

    let result = if cache.inputs_names.is_some() {
        let list = group_inputs_list_for_path(&path, project_cfg, global_cfg, cache)?;
        list.len()
    } else {
        let mut count = 0usize;
        if path.eq_ignore_ascii_case("All") {
            count = cache.dat_names.len() + cache.idx_list.len();
        } else if path.eq_ignore_ascii_case("Inputs") {
            count = cache.dat_names.len();
        } else if path
            .get(0..5)
            .map(|s| s.eq_ignore_ascii_case("Input"))
            .unwrap_or(false)
        {
            count = 1;
        } else if let Some(parsed) = parse_group_path(&path, global_cfg, project_cfg) {
            let mut ind_list = Vec::new();
            group_indicators_recursive(&mut ind_list, parsed.conf, parsed.collect_type);
            let ind_set: HashSet<String> =
                ind_list.iter().map(|s| s.to_lowercase()).collect();
            for entry in &cache.idx_list {
                if !parsed.allowed_data.is_empty()
                    && !parsed.allowed_data.iter().any(|d| *d == entry.data_index)
                {
                    continue;
                }
                if let Some(name) = cache.inp_list.get(entry.inp_index) {
                    if ind_set.contains(&name.to_lowercase()) {
                        count += 1;
                    }
                }
            }
        }
        count
    };

    if result == 0 {
        return Err(GggpError::InvalidData(format!(
            "group '{}' has no dataseries",
            group_name
        )));
    }
    group_cache.insert(key, result);
    Ok(result)
}

fn group_inputs_list_for_path(
    path: &str,
    project_cfg: &Node,
    global_cfg: &Node,
    cache: &IndicatorCache,
) -> Result<Vec<i32>, GggpError> {
    let mut list = Vec::new();
    if path.eq_ignore_ascii_case("All") {
        if let Some(inputs_names) = cache.inputs_names.as_ref() {
            for (idx, name) in inputs_names.iter().enumerate() {
                if !name.is_empty() {
                    list.push(idx as i32);
                }
            }
        } else {
            for idx in 0..cache.dat_names.len() {
                list.push(idx as i32);
            }
            for idx in 0..cache.idx_list.len() {
                list.push((cache.dat_names.len() + idx) as i32);
            }
        }
        return Ok(list);
    }
    if path.eq_ignore_ascii_case("Inputs") {
        for idx in 0..cache.dat_names.len() {
            list.push(idx as i32);
        }
        return Ok(list);
    }
    if path
        .get(0..5)
        .map(|s| s.eq_ignore_ascii_case("Input"))
        .unwrap_or(false)
    {
        if let Ok(value) = path[5..].trim().parse::<usize>() {
            if value == 0 || value > cache.dat_names.len() {
                return Err(GggpError::InvalidData(format!(
                    "input index '{}' is out of range",
                    path
                )));
            }
            list.push((value - 1) as i32);
            return Ok(list);
        }
    }

    let parsed = parse_group_path(path, global_cfg, project_cfg)
        .ok_or_else(|| GggpError::InvalidData("group not found".to_string()))?;
    let mut ind_list = Vec::new();
    group_indicators_recursive(&mut ind_list, parsed.conf, parsed.collect_type);
    let ind_set: HashSet<String> = ind_list.iter().map(|s| s.to_lowercase()).collect();
    let mut entries: Vec<(usize, &IndicatorEntry)> =
        cache.idx_list.iter().enumerate().collect();
    entries.sort_by_key(|(idx, entry)| {
        entry
            .input_index
            .unwrap_or(cache.dat_names.len() + *idx)
    });
    for (idx, entry) in entries {
        if !parsed.allowed_data.is_empty()
            && !parsed.allowed_data.iter().any(|d| *d == entry.data_index)
        {
            continue;
        }
        let name = &entry.paramless_path;
        if ind_set.contains(&name.to_lowercase()) {
            let value = entry
                .input_index
                .unwrap_or(cache.dat_names.len() + idx) as i32;
            list.push(value);
        }
    }

    if list.is_empty() {
        return Err(GggpError::InvalidData(format!(
            "group '{}' has no dataseries",
            path
        )));
    }
    Ok(list)
}

fn index_of_indicator(name: &str, cache: &IndicatorCache) -> Result<usize, GggpError> {
    if let Some(map) = cache.input_name_to_index.as_ref() {
        if let Some(index) = map.get(&name.to_lowercase()) {
            return Ok(*index);
        }
    }
    for (idx, dat) in cache.dat_names.iter().enumerate() {
        if dat.eq_ignore_ascii_case(name) {
            return Ok(idx);
        }
    }
    for (idx, entry) in cache.idx_list.iter().enumerate() {
        let full = full_indicator_name(entry);
        if full.eq_ignore_ascii_case(name) {
            return Ok(cache.dat_names.len() + idx);
        }
    }
    Err(GggpError::InvalidData(format!(
        "indicator '{}' is not defined",
        name
    )))
}

fn full_indicator_name(entry: &IndicatorEntry) -> String {
    if entry.full_path.is_empty() {
        format!("DATA{}", entry.data_index)
    } else {
        format!("DATA{}>{}", entry.data_index, entry.full_path)
    }
}

fn parse_data_index(name: &str) -> i32 {
    if let Some(prefix) = name.get(0..4) {
        if prefix.eq_ignore_ascii_case("DATA") {
            if let Ok(value) = name[4..].trim().parse::<i32>() {
                return value;
            }
        }
    }
    0
}

fn group_index(name: &str, data: &GpData) -> Result<usize, GggpError> {
    for (idx, group_name) in data.groups_names.iter().enumerate() {
        if group_name.eq_ignore_ascii_case(name) {
            return Ok(idx);
        }
    }
    Err(GggpError::InvalidConfig("group not found"))
}

fn find_substring(text: &str, pattern: &str, start: usize) -> Option<usize> {
    text[start..].find(pattern).map(|idx| start + idx)
}

const FITNESS_REC_SIZE: usize = 16;

fn individuals_size(individuals: &[GpIndividual]) -> usize {
    let mut size = 4;
    for ind in individuals {
        size += ind.size();
    }
    size
}

fn fitness_history_size(history: &HashMap<u32, FitnessHistoryRec>) -> usize {
    8 + history.len() * (4 + FITNESS_REC_SIZE)
}

fn write_fitness_history<W: Write>(
    writer: &mut W,
    history: &HashMap<u32, FitnessHistoryRec>,
) -> io::Result<()> {
    write_i32(writer, history.len() as i32)?;
    write_i32(writer, FITNESS_REC_SIZE as i32)?;
    let mut keys: Vec<u32> = history.keys().copied().collect();
    keys.sort_by_key(|k| *k as i32);
    for key in keys {
        let rec = &history[&key];
        write_u32(writer, key)?;
        let mut buf = vec![0u8; FITNESS_REC_SIZE];
        buf[0] = if rec.success { 1 } else { 0 };
        if FITNESS_REC_SIZE >= 16 {
            buf[8..16].copy_from_slice(&rec.fitness.to_le_bytes());
        }
        writer.write_all(&buf)?;
    }
    Ok(())
}

fn read_fitness_history<R: Read>(
    reader: &mut R,
) -> Result<HashMap<u32, FitnessHistoryRec>, GggpError> {
    let count = read_i32(reader)?;
    let rec_size = read_i32(reader)?;
    if count < 0 || rec_size < 0 {
        return Err(GggpError::InvalidData(
            "negative fitness history header".to_string(),
        ));
    }
    let rec_size = rec_size as usize;
    let mut history = HashMap::new();
    for _ in 0..count {
        let key = read_u32(reader)?;
        let raw = read_bytes(reader, rec_size)?;
        let mut buf = vec![0u8; FITNESS_REC_SIZE];
        let copy_len = rec_size.min(FITNESS_REC_SIZE);
        buf[..copy_len].copy_from_slice(&raw[..copy_len]);
        let success = buf.get(0).copied().unwrap_or(0) != 0;
        let mut fit_bytes = [0u8; 8];
        if FITNESS_REC_SIZE >= 16 {
            fit_bytes.copy_from_slice(&buf[8..16]);
        }
        let fitness = f64::from_le_bytes(fit_bytes);
        history.insert(
            key,
            FitnessHistoryRec {
                success,
                fitness,
            },
        );
    }
    Ok(history)
}

fn write_i32<W: Write>(writer: &mut W, value: i32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_bool<W: Write>(writer: &mut W, value: bool) -> io::Result<()> {
    writer.write_all(&[if value { 1 } else { 0 }])
}

fn write_f64<W: Write>(writer: &mut W, value: f64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i32<R: Read>(reader: &mut R) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_bool<R: Read>(reader: &mut R) -> io::Result<bool> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

fn read_f64<R: Read>(reader: &mut R) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bytes<R: Read>(reader: &mut R, len: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

impl Phenotype for GpTree {
    fn render(&self) -> OutputType {
        OutputType::SourceCode(self.indented_text(0))
    }

    fn dimension(&self) -> usize {
        1
    }
}

impl VectorPhenotype for GpTree {
    fn render_vector(&self, dim: usize) -> nalgebra::DVector<f64> {
        compile_tree_to_vector(self, dim, None)
    }

    fn to_vector_symbol(&self, _gene_index: usize, _depth: i32, _parent_hash: u64) -> Option<VectorSymbol> {
        None
    }
}

