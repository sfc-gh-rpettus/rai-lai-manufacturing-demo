import asyncio
import io
import logging
import os
import re
import sys
import threading
import uuid
import warnings
import requests
import queue

import streamlit as st

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

from agent_gateway import Agent
from agent_gateway.tools import PythonTool
from snowflake.cortex import Complete

# App title and introduction
st.title("🏭 Manufacturing Agent")
st.markdown("""
### Computer Vision with LandingAI + Graph Reasoning with RelationalAI
""")
st.write("This is under active development.")

# Get the current Snowflake session
session = get_active_session()

tab1, tab2 = st.tabs(["🗒 Instructions", "🤖 Chat with Agent"])

# -------------------------------
# Defining functions (tools)
# -------------------------------

def landing_lens_inference(file):
    """Tool to run defect inferencing on a file using Landing Lens"""
    logging.info("Running landing_lens_inference tool")
    sql_cmd = f"""
    SELECT
    LANDINGLENS__VISUAL_AI_PLATFORM_NON_COMMERCIAL_USE.core.run_inference('{file}'
        , '12987d0c-6915-4b79-9e25-cabecd2fe85f') as inference
    """
    inference_out = session.sql(sql_cmd).to_pandas()['INFERENCE'][0]
    return inference_out

def landing_lens_summarize(input):
    """Summarize inferencing"""
    logging.info("Running landing_lens_summarize tool")
    return Complete(model, f"#INSTRUCTIONS: Summarize if a defect was found based on the following contents. BE SURE TO INCLUDE THE FILE NAME! #CONTENTS:  {input}")

def defect_recommendation(input):
    """Create a summary including the results of the batch analysis"""
    logging.info("Running defect_recommendation tool")
    return Complete(model, f"#INSTRUCTIONS: Provide an impact analysis of this part if defective based on the batch count impact. If the count exceeds 20 say this needs to be FLAGGED!!!! {input}")

def query_part():
    """Query defective images to find the part number and operator"""
    logging.info("Running query_part tool")
    sql_cmd = f"""
    select * 
    from RAI_DEMO.RAI_LAI_MANUFACTURING.MANUFACTURING_HISTORY 
    where REGEXP_SUBSTR(FILENAME, '[^/]+/[^/]+$') = '{selected_image}' limit 1
    """
    df = session.sql(sql_cmd).to_pandas()
    return df.to_dict()

def query_casting_batch(casting_batch_id):
    """Query to find all items associated with a casting batch id"""
    logging.info("Running query_casting_batch tool")
    sql = f"""
        select count(*) from RAI_DEMO.RAI_LAI_MANUFACTURING.MANUFACTURING_HISTORY 
        where casting_batch_id = '{casting_batch_id}'
        and label_name = 'defects'
        """
    df = session.sql(sql).to_pandas()
    return df.to_dict()

def key_value_extractor(query_part_output):
    """Extract out the value associated with a key from a dictionary"""
    logging.info("Running key_value_extractor tool")
    return Complete(model, f"#INSTRUCTIONS: You are to extract the casting_batch_id value from the following. DO NOT INLCUDE ANYTHING BUT THE value!!! #INPUT: {query_part_output}")

# -------------------------------
# Configure and instantiate tools
# -------------------------------

ll_config = {
    "tool_description": "produces a classification for defects given an image",
    "output_description": "dictionary of defect prediction metadata",
    "python_func": landing_lens_inference,
}

ll_summarize_config = {
    "tool_description": "summarizes whether a defect is present given a json classification output",
    "output_description": "string of summarized defect findings",
    "python_func": landing_lens_summarize,
}

defect_recommendation_config = {
    "tool_description": "Provide an impact assessment for the part ONLY FOR defective parts. You will need to use the casting_batch_tool first and include results from other tools",
    "output_description": "string of summarized analysis",
    "python_func": defect_recommendation,
}

query_part_config = {
    "tool_description": "query to find the part number, operator, and batch_id for the given image. You will use this to find the batch id associated with an image",
    "output_description": "dict of summary data",
    "python_func": query_part,
}

casting_batch_config = {
    "tool_description": "find count of items for a batch id. You will need to make sure to use the key_value_extract first to get the casting_batch_id value",
    "output_description": "dict of data",
    "python_func": query_casting_batch,
}

key_value_extract = {
    "tool_description": "finds the value associated with a given key from the part output",
    "output_description": "dict of data",
    "python_func": key_value_extractor,
}

ll_inference_tool = PythonTool(**ll_config)
ll_summarize_tool = PythonTool(**ll_summarize_config)
defect_recommendation_tool = PythonTool(**defect_recommendation_config)
query_part_tool = PythonTool(**query_part_config)
casting_batch_tool = PythonTool(**casting_batch_config)
key_value_extract_tool = PythonTool(**key_value_extract)

snowflake_tools = [
    ll_inference_tool, 
    ll_summarize_tool, 
    defect_recommendation_tool, 
    query_part_tool, 
    casting_batch_tool,
    key_value_extract_tool
]

# Create the agent
agent = Agent(snowflake_connection=session, tools=snowflake_tools, max_retries=5)

with tab1:
    st.markdown("""

    ### Instructions
    This app demonstrates the art of the possible by leveraging the Agent framework to orchestrate tasks using Snowflake Native Apps as tools.
    The app leverages the [orchestration-framework](https://github.com/Snowflake-Labs/orchestration-framework) developed by SIT.

    This premise is to show how an agent can reason in a manufacturing setting to identify defects on a part and to analyze the impact on the supply chain.

    - 🖼 **LandingAI** is used as a tool for defect classification
    - 🧠 **RelationalAI** is used as a tool for graph analysis for supply chain impacts (COMING SOON!!!)
    - 🛠️ Additional Python tools are used to enable insights and to aid in agentic reasoning


    #### Example Questions to Ask
    **Basic Questions**
    - What is the batch id associated with this part?
    - Who was the operator on this part?

    **Landing AI Question**
    - Is this part defective?
    - What is the defect score of this part?
    - Provide a summary of the defect?

    **Advanced Questions**
    - Provide an insight analysis for this part
    - Is this part defective? What is the impact if so?
    
    """
               )

with tab2:

    # -------------------------------
    # User Selections: Model & Image
    # -------------------------------
    
    model = st.selectbox("Choose a model", 
                         [
                             'claude-3-5-sonnet',
                             'snowflake-arctic',
                             'mistral-large',
                             'reka-flash',
                             'llama2-70b-chat', 
                             'mixtral-8x7b', 
                             'mistral-7b',
                         ]
    )
    
    selected_image = st.selectbox("Select an image",
                                 [
                                     'defects/cast_def_0_0.jpeg',
                                     'defects/cast_def_0_105.jpeg',
                                     'defects/cast_def_0_107.jpeg',
                                     'defects/cast_def_0_11.jpeg',
                                     'defects/cast_def_0_110.jpeg',
                                     'defects/cast_def_0_118.jpeg',
                                     'defects/cast_def_0_133.jpeg',
                                     'defects/cast_def_0_144.jpeg',
                                     'defects/cast_def_0_148.jpeg',
                                     'no-defects/cast_ok_0_102.jpeg'
                                 ]
    )
    
    # Build the file path using the selected image
    file = f'@llens_sample_ds_manufacturing.ball_bearing.dataset/data/{selected_image}'
    
    # Retrieve and display the image with a spinner
    with st.spinner("Loading image..."):
        image_bytes = session.file.get_stream(file, decompress=False).read()
        st.image(image_bytes, caption=selected_image)
    
    # -------------------------------
    # Helper function to extract tool name from log messages
    # -------------------------------
    
    def extract_tool_name(statement):
        # Assuming the log message format is "Running <tool_name> tool"
        match = re.search(r"Running\s+(.*?)\s+tool", statement)
        if match:
            return match.group(1)
        return "Unknown Tool"
    
    # -------------------------------
    # Setup logging handler to capture tool usage messages
    # -------------------------------
    
    class QueueLogHandler(logging.Handler):
        def __init__(self, log_queue):
            super().__init__()
            self.log_queue = log_queue
        def emit(self, record):
            msg = self.format(record)
            if "Running" in msg and "tool" in msg:
                tool_name = extract_tool_name(msg)
                self.log_queue.put({"tool": tool_name})
    
    log_queue = queue.Queue()
    queue_handler = QueueLogHandler(log_queue)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    queue_handler.setFormatter(formatter)
    logging.getLogger().addHandler(queue_handler)
    logging.getLogger().setLevel(logging.INFO)
    
    # -------------------------------
    # Conversation Interface with Tool Tracking & Cleaned Output
    # -------------------------------
    
    # Reset conversation if requested
    if st.button("Start Over"):
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
        st.rerun()
    
    # Initialize conversation history if not already set
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help?"}]
    
    st.subheader("Conversation with Manufacturing Agent")
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input("Type your message")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Placeholders for tool updates and response
        tool_container = st.empty()
        msg_queue = queue.Queue()
    
        def run_agent_acall():
            # Capture printed output from the agent call
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(agent.acall(f"{prompt} .Use this file: '{file}'"))
            loop.close()
            sys.stdout = old_stdout
    
            # Process any printed output for tool usage messages
            output = new_stdout.getvalue()
            for line in output.splitlines():
                if "Running" in line and "tool" in line:
                    tool_name = extract_tool_name(line)
                    msg_queue.put({"tool": tool_name})
            # Finally, push the final response
            msg_queue.put({"response": response})
    
        # start the agent call in a separate thread
        thread = threading.Thread(target=run_agent_acall)
        thread.start()
    
        with st.chat_message("assistant"), st.spinner("Awaiting Response..."):
            final_response = None
            while True:
                try:
                    # Check messages captured from the agent thread
                    while not msg_queue.empty():
                        msg = msg_queue.get_nowait()
                        if "tool" in msg:
                            tool_container.markdown(f"**Using tool:** {msg['tool']}")
                        elif "response" in msg:
                            final_response = msg["response"]
                            break
                    # Also check logging messages from the log queue
                    while not log_queue.empty():
                        log_msg = log_queue.get_nowait()
                        tool_container.markdown(f"**Using tool (log):** {log_msg['tool']}")
                    if final_response is not None:
                        break
                except queue.Empty:
                    continue
            thread.join()
            tool_container.empty()
    
            # clean up the final output
            if isinstance(final_response, dict):
                output_text = final_response.get("output", "")
                sources = final_response.get("sources", [])
                tool_names = [src.get("tool_name") for src in sources if src.get("tool_name")]
                tools_str = ", ".join(tool_names)
                st.markdown(f"**Final Output:** {output_text}")
                st.markdown(f"**Tools Used:** {tools_str}")
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            else:
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
