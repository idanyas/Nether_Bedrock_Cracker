use std::sync::Arc;
use async_std::fs;
use crate::tab::bedrock::block_entry::{Block, BlockMessage};
use crate::tab::controls::{ApplicationTab, CrackerEvent, CrackerState, TabMessage};

use async_std::fs::File;
use async_std::task::spawn_blocking;
use iced::futures::io::BufWriter;
use iced::futures::{AsyncWriteExt, SinkExt};
use iced::{futures, Element, Length, Padding, Subscription, subscription, Command};
use bedrock_cracker::{CrackProgress, estimate_result_amount, search_bedrock_pattern, search_bedrock_pattern_with_list};
use bedrock_cracker::raw_data::block::Block as BlockInfo;

use iced::widget::{button, Column, column, pick_list, row, Scrollable, text, tooltip, checkbox};
use iced::widget::tooltip::Position;
use rfd::AsyncFileDialog;
use tokio::sync::mpsc::channel;
use bedrock_cracker::raw_data::block_type::BlockType;
use bedrock_cracker::raw_data::modes::{BedrockGeneration, OutputMode};
use std::time::Instant;

#[derive(Debug, Default)]
pub struct BdrkTab {
    estimated_seeds: u64,
    blocks: Vec<Block>,
    valid_blocks: Vec<BlockInfo>,
    mode: BedrockGeneration,
    output_mode: OutputMode,
    seed_list: Arc<Vec<u64>>,
    use_gpu: bool,
}

#[derive(Debug, Clone)]
pub enum BdrkMessage {
    Block(usize, BlockMessage),
    CrackerMode(BedrockGeneration),
    OutputMode(OutputMode),
    LoadSeedList,
    LoadedSeedList(Option<String>),
    ToggleGpu(bool),
}

impl From<TabMessage> for BdrkMessage {
    fn from(tab_msg: TabMessage) -> Self {
        match tab_msg {
            TabMessage::BdrkMessage(msg) => msg,
        }
    }
}

impl ApplicationTab for BdrkTab {
    type Message = BdrkMessage;

    fn new() -> Self {
        Self {
            estimated_seeds: (1 << 48),
            blocks: vec![Block::new()],
            valid_blocks: Vec::new(),
            mode: BedrockGeneration::Normal,
            output_mode: OutputMode::WorldSeed,
            seed_list: Arc::new(Vec::new()),
            use_gpu: false,
        }
    }

    fn load_config(&mut self, file: String) {
        println!("[GUI] Loading config...");
        let mut blocks = vec![];
        for line in file.lines().take(200) {
            let block = Block::from(line);
            blocks.push(block);
        }
        self.blocks = blocks;
        self.update_blocks();
        println!("[GUI] Config loaded with {} blocks.", self.blocks.len());
    }

    fn save_config(&self) -> String {
        println!("[GUI] Saving config with {} valid blocks.", self.valid_blocks.len());
        let mut content = String::new();
        for block in self.valid_blocks.iter() {
            content.push_str(&format!("{}\n", block))
        }
        content
    }

    fn update(&mut self, message: Self::Message) -> Command<TabMessage> {
        match message {
            BdrkMessage::Block(index, BlockMessage::Deleted) => {
                self.blocks.remove(index);
                self.update_blocks();
            }

            BdrkMessage::Block(index, message) => {
                if let Some(block) = self.blocks.get_mut(index) {
                    block.update(message);
                    self.update_blocks();
                }
            }
            BdrkMessage::CrackerMode(mode) => {
                println!("[GUI] Cracker mode changed to: {}", mode);
                self.mode = mode;
                self.update_blocks()
            }
            BdrkMessage::OutputMode(mode) => {
                println!("[GUI] Output mode changed to: {}", mode);
                self.output_mode = mode;
            }
            BdrkMessage::LoadSeedList => {
                return Command::perform(
                    async {
                        let handle = AsyncFileDialog::new().pick_file().await?;
                        let path = handle.path().to_owned();
                        let file = fs::read_to_string(&path).await.ok()?;
                        Some((file, path.to_string_lossy().to_string()))
                    },
                    |res| {
                        if let Some((content, path)) = res {
                            println!("[GUI] Loaded seed list from: {}", path);
                            BdrkMessage::LoadedSeedList(Some(content))
                        } else {
                            BdrkMessage::LoadedSeedList(None)
                        }
                    }
                ).map(TabMessage::BdrkMessage);
            }
            BdrkMessage::LoadedSeedList(seed_file) => {
                if let Some(seeds) = seed_file {
                    let seed_list: Vec<_> = seeds.lines().filter_map(|line| line.parse::<u64>().ok()).collect();
                    println!("[GUI] Parsed {} seeds from list.", seed_list.len());
                    self.seed_list = Arc::new(seed_list);
                } else {
                    println!("[GUI] Unloaded seed list.");
                    self.seed_list = Arc::new(Vec::new());
                }
            }
            BdrkMessage::ToggleGpu(val) => {
                println!("[GUI] GPU Usage toggled: {}", val);
                self.use_gpu = val;
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, TabMessage> {
        let estimate = text(format!(
            "Naively estimated results: {} seeds",
            self.estimated_seeds
        ))
            .width(Length::Fill);
        let crack_mode = pick_list(
            &BedrockGeneration::ALL[..],
            Some(self.mode),
            BdrkMessage::CrackerMode,
        );
        let output_mode = pick_list(
            &OutputMode::ALL[..],
            Some(self.output_mode),
            BdrkMessage::OutputMode,
        );
        let seed_list_button: Element<_> = if self.seed_list.is_empty() {
            Element::from(tooltip(
                button("Load seed list").on_press(BdrkMessage::LoadSeedList),
                "Check a text file of structure seeds against the bedrock positions",
                Position::Bottom
            ))
        } else {
            Element::from(
                button("Unload seed list").on_press(BdrkMessage::LoadedSeedList(None))
            )
        };
        
        // GPU Checkbox - builder pattern
        let gpu_toggle = checkbox("Use GPU", self.use_gpu)
            .on_toggle(BdrkMessage::ToggleGpu);

        let top_bar = row![estimate, crack_mode, output_mode, seed_list_button, gpu_toggle].spacing(10);
        let coords: Element<_> = column(
            self.blocks
                .iter()
                .enumerate()
                .map(|(i, task)| {
                    task.view(i == self.blocks.len() - 1)
                        .map(move |message| BdrkMessage::Block(i, message))
                })
                .collect::<Vec<_>>(),
        )
        .padding(Padding::from([5, 20]))
        .spacing(5)
        .into();
        let coords = Scrollable::new(coords).height(Length::Fill);
        let view: Element<_> = Column::with_children(vec![top_bar.into(), coords.into()]).into();
        view.map(TabMessage::BdrkMessage)
    }

    fn poll_cracker(&self, state: &CrackerState, threads: &str) -> Subscription<CrackerEvent> {
        match &state {
            CrackerState::Idle => Subscription::none(),
            CrackerState::Starting(file_output) => {
                println!("[GUI] Cracker Starting...");
                let threads = threads.parse::<u64>().unwrap_or(1);

                crack(&self.valid_blocks, &self.seed_list, file_output, threads, self.mode, self.output_mode, self.use_gpu)
            }
            CrackerState::Running => subscription::run_with_id(
                std::any::TypeId::of::<Unique>(),
                futures::stream::pending(),
            ),
        }
    }
}

impl BdrkTab {
    fn update_blocks(&mut self) {
        self.add_entry();
        self.update_invalid_states();
        self.estimated_seeds = estimate_result_amount(&self.valid_blocks).max(1);
    }

    fn add_entry(&mut self) {
        if let Some(last_block) = self.blocks.last() {
            if !last_block.is_empty() {
                self.blocks.push(Block::new())
            }
        } else {
            self.blocks.push(Block::new())
        }
    }

    /// check for multiple blocks in the same position etc...
    fn update_invalid_states(&mut self) {
        let mut valid_blocks: Vec<BlockInfo> = vec![];
        for gui_block in self.blocks.iter_mut() {
            if let Some(block) = gui_block.is_valid_pos() {
                let invalid = Self::check_invalid(&block, &valid_blocks, self.mode);
                if !invalid {
                    valid_blocks.push(block);
                }
                gui_block.set_duplicate(invalid);
            }
        }
        self.valid_blocks = valid_blocks;
    }

    fn check_invalid(block: &BlockInfo, valid_blocks: &[BlockInfo], mode: BedrockGeneration) -> bool {
        for valid_block in valid_blocks.iter() {
            if block.x == valid_block.x &&
                block.z == valid_block.z &&
                (block.y > 5) == (valid_block.y > 5)
            {
                if block.y == valid_block.y { return true }
                if mode == BedrockGeneration::Paper1_18 {
                    if block.block_type == valid_block.block_type { return true }
                    let mut y1 = valid_block.y;
                    let mut y2 = block.y;
                    if (y1 > 5) ^ (block.block_type == BlockType::OTHER) {
                        (y1, y2) = (y2, y1);
                    }
                    if y1 <= y2 { return true }
                }
            }
        }
        false
    }
}

struct Unique;

pub fn crack(
    blocks: &Vec<BlockInfo>,
    seed_list: &Vec<u64>,
    file_output: &Option<String>,
    threads: u64,
    mode: BedrockGeneration,
    output_mode: OutputMode,
    use_gpu: bool,
) -> Subscription<CrackerEvent> {
    let file_output = file_output.clone();
    let blocks: Vec<_> = blocks.clone();
    let seed_list = seed_list.clone();

    subscription::channel(std::any::TypeId::of::<Unique>(), 100, move |mut output| {
        let file_output = file_output.clone();
        let blocks = blocks.clone();
        let seed_list = seed_list.clone();
        
        async move {
            let mut writer: Option<BufWriter<File>> = create_file_writer(&file_output).await;

            output
                .send(CrackerEvent::Started)
                .await
                .expect("TODO: panic message");

            let (sender, mut receiver) = channel(100);

            // Pre-calculate Total Seeds and Flags BEFORE move
            let total_seeds: u64 = if seed_list.is_empty() { 1u64 << 48 } else { seed_list.len() as u64 };
            let use_list = !seed_list.is_empty();

            if use_list {
                spawn_blocking(move || search_bedrock_pattern_with_list(&blocks, threads, &seed_list, mode, sender));
            } else {
                spawn_blocking(move || search_bedrock_pattern(&blocks, threads, mode, output_mode, sender, use_gpu));
            }

            let mut seeds = vec![];
            
            // Progress tracking variables
            let start_time = Instant::now();
            let mut last_log_time = Instant::now();
            let mut seeds_checked: u64 = 0;
            let mut total_found: usize = 0;

            while let Some(pl_event) = receiver.recv().await {
                match pl_event {
                    CrackProgress::Progress(num) => {
                        seeds_checked += num;
                        let percentage = seeds_checked as f32 / total_seeds as f32;
                        
                        // Terminal Logging for Progress (approx every 1s)
                        if last_log_time.elapsed().as_secs() >= 1 {
                            let elapsed = start_time.elapsed().as_secs_f64();
                            let speed = seeds_checked as f64 / elapsed; // seeds per second
                            let speed_m = speed / 1_000_000.0;
                            
                            let eta_seconds = if speed > 0.0 {
                                (total_seeds.saturating_sub(seeds_checked)) as f64 / speed
                            } else {
                                0.0
                            };
                            
                            let eta_str = if eta_seconds > 3600.0 {
                                format!("{:.1}h", eta_seconds / 3600.0)
                            } else if eta_seconds > 60.0 {
                                format!("{:.1}m", eta_seconds / 60.0)
                            } else {
                                format!("{:.0}s", eta_seconds)
                            };

                            println!("[Progress] {:.2}% | {:.2} MSeeds/s | ETA: {} | Found: {}", 
                                percentage * 100.0, speed_m, eta_str, total_found);
                            
                            last_log_time = Instant::now();
                        }

                        let update = CrackerEvent::ProgressUpdate(percentage, seeds);
                        seeds = vec![];
                        output.send(update).await.unwrap();
                    }
                    CrackProgress::Seed(num) => {
                        total_found += 1;
                        println!("[Found] Seed: {}", num);
                        let seed = (num as i64).to_string();
                        if let Some(ref mut writer) = writer {
                            let line = format!("{}\n", seed);
                            writer.write_all(line.as_bytes()).await.unwrap();
                        }
                        seeds.push(seed);
                    }
                };
            }
            let elapsed = start_time.elapsed().as_secs_f64();
            println!("==================================================");
            println!(" Search Finished.");
            println!(" Total Time: {:.2}s", elapsed);
            println!(" Total Seeds Checked: {}", seeds_checked);
            println!(" Total Matches Found: {}", total_found);
            println!("==================================================");
            
            output.send(CrackerEvent::Finished).await.unwrap();

            iced::futures::future::pending().await
        }
    })
}

async fn create_file_writer(file: &Option<String>) -> Option<BufWriter<File>> {
    match file {
        None => None,
        Some(file) => {
            println!("[GUI] Creating output file: {}", file);
            let file = File::create(file).await.ok()?;
            Some(BufWriter::new(file))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_invalid() {
        let block = BlockInfo::new(1,1,1,BlockType::BEDROCK);
        let mut valid_blocks = vec![
            BlockInfo::new(1,2,1,BlockType::BEDROCK),
            BlockInfo::new(1,3,1,BlockType::BEDROCK)
        ];
        //no duplicates -> block is valid
        assert!(!BdrkTab::check_invalid(&block, &valid_blocks, BedrockGeneration::Normal));

        //duplicates -> block is invalid
        valid_blocks.push(block.clone());
        assert!(BdrkTab::check_invalid(&block, &valid_blocks, BedrockGeneration::Normal));
    }

    #[test]
    fn test_check_invalid_paper() {
        let mut block = BlockInfo::new(1, 1, 1, BlockType::BEDROCK);
        let valid_blocks = vec![
            BlockInfo::new(1,2,1,BlockType::OTHER)
        ];
        //valid position
        assert!(!BdrkTab::check_invalid(&block, &valid_blocks, BedrockGeneration::Paper1_18));

        //bedrock ont op of other is an invalid placement
        block.y = 3;
        assert!(BdrkTab::check_invalid(&block, &valid_blocks, BedrockGeneration::Paper1_18));

        //Two of the same type in the same column is redundant
        block.block_type = BlockType::OTHER;
        assert!(BdrkTab::check_invalid(&block, &valid_blocks, BedrockGeneration::Paper1_18));

        //valid on opposite sites
        block.y = 123;
        assert!(!BdrkTab::check_invalid(&block, &valid_blocks, BedrockGeneration::Paper1_18));
    }
}
