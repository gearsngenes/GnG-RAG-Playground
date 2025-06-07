def full_prompt_phi4(context, history, query):
        return (
                f"Respond to the user's latest USER QUERY using and citing ONLY "
                f"the information in the CONTEXT section as reference. Your "
                f"response MUST FOLLOW ALL of the below rules:\n\n"
                
                "- Your total response length must be **500 words or less**.\n"
                "- Cite ONLY the sources you ACTUALLY USE from CONTEXT using the markdown links "
                "provided for each source. The format for inline citation are shown here:\n"
                "  - For documents: [filename](url)\n"
                "  - For images: use ![image description](url) on its own line.\n"
                "- Do not cite any source from CONTEXT more than once."
                "---\n\n"
                
                "###CONTEXT\n"
                f"{context}\n\n"
                
                "---\n\n"
                
                "###USER QUERY\n"
                f"{history}\n"
                f"LATEST QUERY: {query}"
        )

def general_knowledge_prompt(history, query):
        return (
                "Please respond to the user's latest query/statement to the best of your ability. "
                "Below is the prior conversation and the latest user statement you must answer:\n\n"
                "---\nCHAT HISTORY\n"
                f"{history}\n\n"
                "---\nUSER QUERY\n"
                f"{query}"
        )

def full_prompt_4o(context, history, query):
        return (
                f"You are a Markdown-ready assistant responding to a conversation.\n\n"
                "Use ONLY the following text chunks and image descriptions to answer the latest user question in the context of the full discussion.\n\n"
                "Each chunk and description has a source markdown link to refer to the original source from when generating a response. Follow these citation rules:\n"
                "- If you use information from a specific source from the context chunks to generate a response, then you MUST:\n"
                "  a) explicitly refer to the name of the source file you are referring to, AND\n"
                "  b) incorporate the corresponding markdown link into the response to provide the user a way to view the source.\n"
                "- You can only use a specific markdown link once, the first time you explicitly mention the source file that the content came from. Do NOT repeatedly display or link to the same file every time.\n"
                "- If the source file is an image description, you MUST use a markdown image embed by placing a `!` in front of the link so the image renders fully.\n"
                "- If the user's query closely matches the description of an image, or implies or directly requests some visual, prioritize matching images to their request and fully rendering them.\n"
                "- Only use information contained within the text chunks and image descriptions — do NOT rely on general knowledge unless it is to fill obvious gaps or best match available information to the user's query.\n\n"
                "---\n"
                "Chunks:\n"
                f"{context}\n\n"
                "---\n"
                "Conversation History:\n"
                f"{history}\n\n"
                "---\n"
                "Respond to the latest user query/statement below:\n"
                f"{query}"
        )
