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
import random

import streamlit as st

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

from agent_gateway import Agent
from agent_gateway.tools import PythonTool
from snowflake.cortex import Complete


sys.path.append("./relationalai.zip")
#imports = ('@rai_demo.public.packages/relationalai.zip')
#sys.path.append('@rai_demo.public.packages/relationalai.zip')

import relationalai as rai
from relationalai.std import alias, aggregates, rel
from relationalai.std.graphs import Graph

# Get the current Snowflake session
session = get_active_session()

# App title and introduction
st.title("🏭 Manufacturing Agent")
st.markdown("""
### Computer Vision with LandingAI + Graph Reasoning with RelationalAI
""")
st.write("This is under active development.")

tab1, tab2, tab3 = st.tabs(["🗒 Instructions", "🤖 Chat with Agent", "RAI Testing"])

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


def get_sku_site():
    """Query defective images to find the site and part"""
    logging.info("Running query_part tool")
    sql_cmd = f"""
    select  sk.name as SKU_NAME, s.name AS SITE_NAME
    from RAI_DEMO.RAI_LAI_MANUFACTURING.IMAGES i
    left join RAI_DEMO.RAI_LAI_MANUFACTURING.SITE s on s.id = i.site
    left join RAI_DEMO.RAI_LAI_MANUFACTURING.SKU sk on sk.id = i.sku
    where i.file_key = '{selected_image}'
    """
    df = session.sql(sql_cmd).to_pandas()
    return df.to_dict()


def key_value_extractor(dict_output):
    """Extract out the value associated with a key from a dictionary"""
    logging.info("Running key_value_extractor tool")
    return Complete(model, f"#INSTRUCTIONS: You are to extract the sku name value from the following. Only include the value. #INPUT: {dict_output}")

def insights_synthesis(input):
    """Create a summary including the results of the batch analysis"""
    logging.info("Running insights_synthesis tool")
    return Complete(model, f"#INSTRUCTIONS: Provide an impact analysis of the impacted sites, locations etc based on the input. {input}")

# -------------------------------
# RAI configuration
# -------------------------------
config = rai.Config({})
config.file_path = "__inline__"

   
#@st.cache_resource()
def get_model():
    model = rai.Model("SUPPLY_CHAIN_DEMO_CARS_WIP", config = config, connection=session, profile="sf-build-demo", isolated=True)
    return model
    

model = get_model()

sf_db = 'RAI_DEMO'
sf_schema = 'RAI_LAI_MANUFACTURING'
sf_namespace= f'{sf_db}.{sf_schema}'


# %%
Site = model.Type("Site", source=f"{sf_namespace}.SITE")
Sku = model.Type("Sku", source=f"{sf_namespace}.SKU")
SkuAtSite = model.Type("SkuAtSite", source=f"{sf_namespace}.SKU_AT_SITE")
BomComponent = model.Type("BomComponent", source=f"{sf_namespace}.BILL_OF_MATERIALS")
BomMaster = model.Type("BomMaster")
Operation = model.Type("Operation", source=f"{sf_namespace}.OPERATION")

#@st.cache_data
def build_supplychain_kg(model, Site, Sku, SkuAtSite, BomComponent, BomMaster, Operation):

    # Add Named Relations between SkuAtSite and SKU / Site
    with model.rule():
        entity = SkuAtSite()
        entity.set(
            for_sku=Sku(id=entity.sku_id),  # SkuAtSite.for_sku --> Sku    
            at_site=Site(id=entity.site_id) # SkuAtSite.at_site --> Site
        ) 

    # Extract BomMaster from BILL_OF_MATERIALS, Connect BomMaster / BomComponent to Sku / Site, 
    with model.rule():
        bomComponent = BomComponent()
        bomComponent.set(my_SkuAtSite = 
            SkuAtSite(sku_id=bomComponent.input_sku_id, site_id=bomComponent.site_id))
        bomMaster = BomMaster.add(my_SkuAtSite =
            SkuAtSite(sku_id=bomComponent.output_sku_id, site_id=bomComponent.site_id))
        # BomMaster.has_components --> BomComponent
        bomMaster.has_components.extend([bomComponent])

    # OPERATION Table --> Operation Entity
    Operation = model.Type("Operation", source=f"{sf_namespace}.OPERATION")
    with model.rule():
        entity = Operation()
        entity.set(output_SkuAtSite= 
                    SkuAtSite(sku_id=entity.output_sku,
                            site_id=entity.output_site_id))
        with model.match():
            with entity.type == "SHIP": # for Shipping
                input_SkuAtSite=SkuAtSite(sku_id=entity.output_sku, site_id=entity.source_site_id)
                entity.input_SkuAtSite.extend([input_SkuAtSite])
            with entity.type == "SUPPLY": # for Shipping
                input_SkuAtSite=SkuAtSite(sku_id=entity.output_sku, site_id=entity.source_site_id)
                entity.input_SkuAtSite.extend([input_SkuAtSite])
            with model.case(): # otherwise (Making or Packing)
                # Link to BOM (transformation) and sourcelocation (transportation)
                entity.set(has_bom=BomMaster(my_SkuAtSite=entity.output_SkuAtSite))
                entity.input_SkuAtSite.extend([entity.has_bom.has_components.my_SkuAtSite])

    static_graph = Graph(model)

    # Project to a Static Supply Chain Graph for visualization and analytics
    static_graph = Graph(model)

    with model.rule(): # SkuAtSite as Nodes
        entity = SkuAtSite()
        static_graph.Node.add(entity)

    with model.rule(): # Operation as Nodes
        entity = Operation()
        static_graph.Node.add(entity)

    with model.rule(): # Input to Operation as Edges
        entity = Operation()
        static_graph.Edge.add(from_=entity.input_SkuAtSite, to=entity)

    with model.rule(): # Operation to Output as Edge
        entity = Operation()
        static_graph.Edge.add(from_=entity, to=entity.output_SkuAtSite)

    static_graph.fetch()

    return static_graph

#@st.cache_data
def get_rai_impact(input): #, site_name):
    logging.info("Running relationalAI query")
    car_supply_chain_graph = build_supplychain_kg(model, Site, Sku, SkuAtSite, BomComponent, BomMaster, Operation)
    # Where Used: raw material impacts in the supply chain 
    with model.query() as select:  # [where_used]
        # find sku with part name
        sku = Sku(name='Brake Disc')
        # find site with vendor name
        site = Site(name='ZF')
        # find all sku/site containging the sku
        start_nodes = SkuAtSite(sku_id=sku.id, site_id=site.id)
        # find all reachable nodes via built-in graph algorithm
        reachable = car_supply_chain_graph.compute.reachable_from(start_nodes)
        # only care about end points of a supply chain: sales regions
        car_supply_chain_graph.compute.outdegree(reachable) == 0
        # return impacted downstream Sku + Site node info
        with SkuAtSite(reachable): 
            result = select(reachable.sku_id, reachable.site_id,
                            alias(reachable.for_sku.name, "Impacted Car"),
                            alias(reachable.at_site.name, "Impacted Region"))

    return result.results.to_dict()

#@st.cache_data
#def get_impact(input): #, site_name):
#    sql_cmd = f"""
#    select  *
#    from RAI_DEMO.RAI_LAI_MANUFACTURING.SKU sk
#    where sk.name = 'Brake Disc' 
#    """
#    result = session.sql(sql_cmd).to_pandas()
#    return result.to_dict()

def get_impact(input):
    result = get_rai_impact("Brake Disc")
    return result

#result = get_impact("Brake Disc", "ZF")
#st.write(get_impact("Brake Disc"))
#st.write(result)  

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

get_sku_site_config = {
    "tool_description": "query to find the sku and site. You will use this to extract the sku name and site name for insights",
    "output_description": "dict of summary data",
    "python_func": get_sku_site,
}

key_value_config = {
    "tool_description": "Extract out the value results",
    "output_description": "dict of data",
    "python_func": key_value_extractor,
}

get_impact_config = {
    "tool_description": "finds the downstream supply chain count impact from a sku name.", # Get the sku name first from the key value config",
    "output_description": "dict of data",
    "python_func": get_impact,
}

insights_synthesis_config = {
    "tool_description": "synthesize your findings of the impact from a dictionary", # Get the sku name first from the key value config",
    "output_description": "string of your summary",
    "python_func": insights_synthesis,
}


ll_inference_tool = PythonTool(**ll_config)
ll_summarize_tool = PythonTool(**ll_summarize_config)
get_sku_site_tool = PythonTool(**get_sku_site_config)
key_value_extract_tool = PythonTool(**key_value_config)
get_impact_tool = PythonTool(**get_impact_config)
insights_synthesis_tool = PythonTool(**insights_synthesis_config)


snowflake_tools = [
    ll_inference_tool, 
    ll_summarize_tool, 
    get_sku_site_tool, 
    #kkey_value_extract_tool,
    get_impact_tool,
    insights_synthesis_tool
    
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
    
    model = 'claude-3-5-sonnet'
    
    selected_image = st.selectbox(
        "Select an image",
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

