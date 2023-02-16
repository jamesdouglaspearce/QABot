class QABot:

    def __init__(self, llm, vectorDB, mem_buffer=10, k=1, relevance_threshold=0.4):
        self.llm = llm
        self.vectorDB = vectorDB
        self.mem_buffer = mem_buffer
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.buffer = []
        self.memory = []
        self.last_doc = ''

    def _update_memory(self, query, response, doc):
        self.memory.append(f'Human: {query}\n')
        self.memory.append(f'SaasyAI: {response}\n')
        self.buffer = self.memory[-self.mem_buffer:]
        self.last_doc = doc

    def query(self, q, buff=None):

        # Perform KNN Vector Search
        # TODO: Add map_reduce for k > 1
        source = self.vectorDB.similarity_search_with_score(q, self.k)
        score = source[0][1]

        # Check relevance of retrieved doc
        # If not very relevant assume still talking about last doc
        if score <= self.relevance_threshold:
            doc = source[0][0].page_content
        else:
            doc = self.last_doc

        # Construct chat history from memory buffer
        if buff is None:
            chat_history = ''.join(self.buffer)
        else:
            chat_history = ''.join(buff[-self.mem_buffer:])

        # Get LLM response from OpenAI
        r = self.llm({'query': q,
                      'source_documents': doc,
                      'chat_history': chat_history}
                     )['text'].strip('\n')

        # Updatre memory for next query
        self._update_memory(q, r, doc)

        return r
