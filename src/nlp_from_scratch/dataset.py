import re

import numpy as np
import numpy.random as npr
import torch
from datasets import load_dataset
from loguru import logger
from tokenizers import Tokenizer
from torch.utils.data.dataset import Dataset

from nlp_from_scratch.constants import (
    CHUNKS_POSITIONS_PATH,
    FREQUENCIES_SAVE_PATH,
    MAX_LEN,
    TOKENIZER_SAVE_PATH,
)
from nlp_from_scratch.utils import get_chunks_from_text


# Function to filter text
def filter_text(text):
    # Keep only alphabet and punctuation
    return re.sub(r"[^\da-zA-Z.,!?;:()'\"]+", " ", text)


class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer: Tokenizer,
        corpus: Dataset,
        frequencies: np.ndarray,
        chunks: list[list[int]],
    ):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.frequencies = frequencies
        self.chunks = chunks

    @property
    def cls_token(self):
        return self.tokenizer.token_to_id("[CLS]")

    @property
    def sep_token(self):
        return self.tokenizer.token_to_id("[SEP]")

    @property
    def mask_token(self):
        return self.tokenizer.token_to_id("[MASK]")

    @property
    def pad_token(self):
        return self.tokenizer.token_to_id("[PAD]")

    def prepare_chunks(self, n_chunks: int = 1000):
        logger.info("Preparing chunks")
        chunks = []
        while len(chunks) < n_chunks:
            k = npr.randint(0, len(self.corpus))
            chunks += get_chunks_from_text(
                self.corpus[k]["text"], self.tokenizer, self.cls_token, self.pad_token
            )
        self.chunks = torch.stack(chunks)
        del chunks
        logger.success(f"OK: {len(self.chunks)} prepared")

    @staticmethod
    def load_from_save(tokenizer_path: str = TOKENIZER_SAVE_PATH):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        corpus = load_dataset("wikipedia", "20220301.en", split="train")
        frequencies = np.load(FREQUENCIES_SAVE_PATH)
        frequencies[:5] = 1.0
        # try:
        #     with open(CHUNKS_POSITIONS_PATH, "r") as f:
        #         chunks = json.load(f)
        # except:
        #     chunks = []
        chunks = []
        return TextDataset(
            tokenizer=tokenizer, corpus=corpus, frequencies=frequencies, chunks=chunks
        )

    @staticmethod
    def load_dummy(tokenizer_path: str = TOKENIZER_SAVE_PATH):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        corpus = [
            {
                "text": "Artificial intelligence (AI) has undergone a remarkable transformation over the past decades. Early AI research in the 1950s focused on symbolic reasoning and rule-based systems, but progress was slow due to computational limitations. The emergence of machine learning in the 1980s and 1990s revolutionized the field, allowing AI to learn from data rather than relying on manually programmed rules. Deep learning, powered by neural networks, further accelerated AI advancements in the 2010s. Today, AI applications range from natural language processing and computer vision to medical diagnostics and autonomous systems. The future of AI is expected to involve more generalizable models capable of reasoning across domains, reducing bias, and improving interpretability. Ethical concerns surrounding AI, such as bias, transparency, and job displacement, continue to be major discussion points. Governments and research institutions are actively working on regulations to ensure AI development aligns with human values. As AI continues to advance, the potential for intelligent automation, personalized recommendations, and scientific breakthroughs grows. However, challenges such as data privacy, adversarial attacks, and ethical AI governance remain critical for sustainable AI deployment. Human decision-making is influenced by a variety of psychological factors, including cognitive biases, emotions, and social influences. One well-known bias is the confirmation bias, where individuals tend to seek information that supports their existing beliefs while ignoring contradictory evidence. The availability heuristic leads people to overestimate the importance of recent or easily recalled events, affecting risk perception. Emotional states also play a role in decision-making; for example, individuals experiencing anxiety may be more risk-averse, while those in a positive mood may make overly optimistic choices. Social factors, such as peer pressure and cultural norms, further shape decision-making processes. In economic contexts, behavioral economics has shown that humans do not always act rationally, as traditional economic models suggest. Instead, people often rely on mental shortcuts, known as heuristics, to simplify complex decisions. Understanding the psychology behind decision-making can improve fields such as marketing, negotiation, and policy-making by accounting for human cognitive tendencies."
            },
            {
                "text": "The invention of the printing press by Johannes Gutenberg in the 15th century revolutionized the dissemination of knowledge. Prior to the printing press, books were laboriously copied by hand, making them expensive and rare. Gutenberg's press used movable type, which allowed for the mass production of books and other printed materials. This innovation dramatically lowered the cost of books, making information more accessible to a broader audience. The spread of printed materials fueled the Renaissance by enabling scholars to share ideas more efficiently. The Protestant Reformation, led by Martin Luther, also benefited from the printing press, as pamphlets and translated Bibles spread religious ideas widely. Over time, newspapers emerged, providing the public with regular updates on political and social issues. The printing press played a crucial role in increasing literacy rates, facilitating scientific discoveries, and shaping modern education. Even today, while digital media has transformed publishing, the legacy of Gutenberg's invention remains foundational to how societies consume and distribute information. The global shift toward renewable energy is driven by concerns over climate change, resource depletion, and energy security. Solar, wind, hydroelectric, and geothermal energy sources offer sustainable alternatives to fossil fuels. Solar energy has seen significant advancements in photovoltaic technology, reducing costs and increasing efficiency. Wind power has also grown, with offshore wind farms providing a steady source of electricity. Hydroelectric power remains a major contributor to global energy production, though it requires careful environmental management to prevent habitat disruption. Geothermal energy, which harnesses heat from the Earth's core, provides a reliable and consistent power source. Despite these advancements, challenges such as energy storage, grid integration, and infrastructure investment remain. Battery technology is improving, enabling better storage solutions for intermittent energy sources like solar and wind. Governments and private companies are investing in research and incentives to accelerate the transition to a cleaner energy future. As technology improves, renewable energy is expected to become the dominant source of global power generation."
            },
            {
                "text": "Space exploration presents numerous challenges, from technological limitations to human health concerns. One of the primary obstacles is the vast distances involved; reaching even the closest celestial bodies, such as Mars, requires extensive planning and months of travel. The harsh conditions of space, including radiation exposure, microgravity, and extreme temperatures, pose significant risks to astronauts. Spacecraft must be designed to withstand these conditions while ensuring the safety and well-being of the crew. Another major challenge is the sustainability of long-duration missions. Supplying astronauts with food, water, and oxygen for extended periods is complex and requires efficient recycling systems. Psychological factors also come into play, as isolation and confinement can impact mental health. Additionally, space agencies must develop propulsion systems capable of reducing travel times and enabling deep-space missions. Advances in artificial intelligence, robotics, and 3D printing are helping to address these challenges, paving the way for future colonization efforts. As humanity continues to explore space, international collaboration and ethical considerations will be crucial in shaping the future of interplanetary travel. Memory formation is a complex process involving different brain regions and neural mechanisms. The hippocampus plays a crucial role in encoding and consolidating memories, while the prefrontal cortex is responsible for retrieving and organizing information. There are different types of memory, including short-term memory, which holds information temporarily, and long-term memory, which stores information for extended periods. Studies have shown that repetition, emotional significance, and context all impact how well information is retained. Sleep is another critical factor in memory consolidation, as the brain strengthens neural connections during deep sleep. Forgetting can occur due to interference from new information, decay over time, or retrieval failures. Neuroplasticity, the brain’s ability to reorganize itself, allows for memory improvement through learning and experience. Understanding the mechanisms of memory formation has implications for treating neurological disorders such as Alzheimer's disease and for developing strategies to enhance learning and retention."
            },
            {
                "text": "Biodiversity plays a critical role in maintaining the stability and resilience of ecosystems. The variety of species within an ecosystem contributes to essential functions such as pollination, nutrient cycling, and climate regulation. When biodiversity declines, ecosystems become more vulnerable to disturbances such as climate change, disease outbreaks, and habitat destruction. For example, the loss of pollinators like bees and butterflies can disrupt food production, leading to decreased crop yields. Similarly, the removal of predators from an ecosystem can result in population imbalances, causing overgrazing and habitat degradation. Conservation efforts aim to protect biodiversity through habitat preservation, reforestation, and sustainable land-use practices. Climate change poses an additional threat, as shifting temperatures and extreme weather events impact species distribution and survival. Protecting biodiversity requires global cooperation, as ecosystems are interconnected and rely on shared resources. By prioritizing conservation and sustainable development, societies can ensure that future generations benefit from the services that healthy ecosystems provide. Artificial intelligence is transforming economies by automating tasks, increasing efficiency, and creating new industries. AI-powered systems are improving productivity in sectors such as healthcare, finance, and manufacturing. In healthcare, AI assists in diagnosing diseases, analyzing medical images, and personalizing treatment plans. In finance, AI-driven algorithms optimize trading strategies, detect fraud, and enhance risk assessment. The rise of AI is also reshaping the labor market, with automation replacing some jobs while creating new opportunities in AI development, data science, and cybersecurity. However, concerns about job displacement and income inequality remain. Governments are exploring policies such as reskilling programs and universal basic income to address these challenges. AI-driven economic growth is expected to increase global GDP, but ethical considerations, such as bias in AI systems and data privacy, must be addressed to ensure equitable benefits."
            },
        ] * 32
        frequencies = np.load(FREQUENCIES_SAVE_PATH)
        frequencies[:5] = 1.0
        return TextDataset(tokenizer=tokenizer, corpus=corpus, frequencies=frequencies)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_index):
        # chunk = self.chunks[chunk_index]
        # text_index = chunk[0]
        # start, end = chunk[1], chunk[2]
        # tokenized = self.tokenizer.encode(filter_text(self.corpus[text_index]["text"])).ids[start:end]
        # return tokenized
        # print(len(tokenized))
        # tokenized_padded = [self.cls_token] + tokenized + (MAX_LEN - len(tokenized) - 1)*[self.pad_token]
        # print(len(tokenized_padded))
        tokens = self.chunks[chunk_index]
        return self.apply_mask(tokens)

    def apply_mask(self, input_ids, mask_prob=0.15):
        labels = input_ids.clone()  # Keep a copy of the original for labels (target)
        attention_mask_vector = labels == self.pad_token
        # attention_mask_matrix = (attention_mask_vector.long().unsqueeze(0).T @ attention_mask_vector.long().unsqueeze(0)).bool()
        # f_tokens = torch.Tensor(self.frequencies[labels])
        # weights = 1 / torch.sqrt(1e-10 + f_tokens)
        # probs = weights / sum(weights) * 0.15 * MAX_LEN
        # Masking process: randomly mask a portion of the tokens
        masked = (
            (torch.rand(MAX_LEN) <= mask_prob)
            * (input_ids != self.cls_token)
            * (input_ids != self.sep_token)
            * (input_ids != self.pad_token)
        )
        input_ids = input_ids * (1 - masked.long()) + masked.long() * self.mask_token
        # for i in range(input_ids.size(0)):
        #     # Skip the [CLS] and [SEP] tokens (index 0 and tokenizer.sep_token_id)
        #     if input_ids[i] == self.cls_token or input_ids[i] == self.sep_token or input_ids[i] == self.pad_token:
        #         continue

        #     # Randomly mask tokens
        #     if npr.rand() < probs[i]:
        #         # Mask with the [MASK] token
        #         input_ids[i] = self.mask_token
        #         mask_tokens[i] = 1

        # Return the modified input_ids (masked) and the labels (original)
        return input_ids, masked, attention_mask_vector, labels
