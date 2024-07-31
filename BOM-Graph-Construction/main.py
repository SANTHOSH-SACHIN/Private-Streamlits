import pandas as pd
import dgl
import torch
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

def read_bom_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    return df

def create_supply_chain_graph(bom_df):
    g = dgl.DGLGraph()
    node_map = {}
    node_features = []
    edge_features = []
    
    # Add dummy parent node
    dummy_parent_id = 'DUMMY_PARENT'
    node_map[dummy_parent_id] = len(node_map)
    g.add_nodes(1)
    dummy_node_features = [0, 0, 0, 0, pd.Timestamp('1970-01-01').timestamp(), 0, 0]
    node_features.append(dummy_node_features)
    
    for i, row in bom_df.iterrows():
        if row['item_id'] not in node_map:
            node_map[row['item_id']] = len(node_map)
            g.add_nodes(1)
            node_features.append([row['lead_time'], row['cost'], row['inventory'], row['min_stock'], 
                                  row['timestamp'].timestamp(), row['transport_cost'], row['transport_time']])
    
    for i, row in bom_df.iterrows():
        if pd.notna(row['parent_id']) and row['parent_id'] in node_map:
            parent_idx = node_map[row['parent_id']]
            child_idx = node_map[row['item_id']]
            g.add_edges(parent_idx, child_idx)
            edge_features.append([row['quantity'], row['assembly_time']])
        elif row['item_id'] in node_map:
            dummy_idx = node_map[dummy_parent_id]
            child_idx = node_map[row['item_id']]
            g.add_edges(dummy_idx, child_idx)
            edge_features.append([0, 0])  # Default edge features for dummy parent

    g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
    g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
    
    return g, node_map





def plot_supply_chain_graph(g, bom_df):
    nx_g = g.to_networkx().to_undirected()
    pos = nx.spring_layout(nx_g, seed=42)
    
    plt.figure(figsize=(15, 10))
    nx.draw(nx_g, pos, with_labels=True, node_size=50, font_size=10, font_color='black', font_weight='bold')
    plt.show()

def inventory_risk_assessment(bom_df):
    for _, row in bom_df.iterrows():
        inventory_ratio = row['inventory'] / row['min_stock']
        timestamp = row["timestamp"]
        if inventory_ratio < 1.2:
            st.markdown(f"\n**Low Inventory Alert [{timestamp}]**: {row['item_name']} (ID: {row['item_id']}) - Current: {row['inventory']}, Minimum: {row['min_stock']}")
        elif inventory_ratio > 3:
            st.markdown(f"\n**Excess Inventory Alert [{timestamp}]**: {row['item_name']} (ID: {row['item_id']}) - Current: {row['inventory']}, Minimum: {row['min_stock']}")

def critical_component_analysis(g, node_map, bom_df):
    nx_g = g.to_networkx()
    betweenness = nx.betweenness_centrality(nx_g)
    critical_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    
    st.subheader("\n\nTop 5 Critical Components:")
    for node_idx, centrality in critical_nodes:
        item_id = [k for k, v in node_map.items() if v == node_idx][0]
        item_name = bom_df[bom_df['item_id'] == item_id]['item_name'].iloc[0]
        st.markdown(f"{item_name} (ID: {item_id}): Centrality = {centrality:.4f}")

def lead_time_analysis(bom_df):
    long_lead_items = bom_df.sort_values('lead_time', ascending=False).head(5)
    st.subheader("\n\nTop 5 Items with Longest Lead Times:")
    for _, row in long_lead_items.iterrows():
        st.markdown(f"{row['item_name']} (ID: {row['item_id']}): {row['lead_time']} days")

def cost_structure_analysis(bom_df):
    total_cost = bom_df['cost'].sum()
    cost_breakdown = bom_df.groupby('item_type')['cost'].sum().sort_values(ascending=False)
    st.subheader("\n\nCost Structure Breakdown:")
    for item_type, cost in cost_breakdown.items():
        percentage = (cost / total_cost) * 100
        st.markdown(f"{item_type}: ${cost:,.2f} ({percentage:.2f}%)")

def supply_chain_risk_analysis(bom_df):
    st.subheader("\n\nSupply Chain Analysis:")
    
    # Supplier concentration
    supplier_concentration = bom_df['supplier'].value_counts()
    st.markdown(f"**Number of unique suppliers**: {supplier_concentration.count()}")
    st.markdown(f"**Top supplier by item count**: {supplier_concentration.index[0]} ({supplier_concentration.iloc[0]} items)")
    
    # Transport method analysis
    transport_method_counts = bom_df['transport_method'].value_counts()
    st.subheader("\nTransport Method Distribution:")
    for method, count in transport_method_counts.items():
        st.markdown(f"{method}: {count} items")
    
    # Long transport time items
    long_transport_items = bom_df.sort_values('transport_time', ascending=False).head(5)
    st.subheader("\nTop 5 Items with Longest Transport Times:")
    for _, row in long_transport_items.iterrows():
        st.markdown(f"{row['item_name']} (ID: {row['item_id']}): {row['transport_time']} days")

import plotly.graph_objects as go

def visualize_graph_plotly(g, node_map, bom_df):
    try:
        # Convert to NetworkX graph
        nx_g = g.to_networkx().to_undirected()
        pos = nx.spring_layout(nx_g)

        # Prepare edge trace (with improved handling for empty/missing data)
        edge_x, edge_y, edge_text = [], [], []
        for edge in nx_g.edges():
            if edge[0] == 'DUMMY_PARENT' or edge[1] == 'DUMMY_PARENT':
                continue  # Skip dummy parent edges
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            parent_id = [k for k, v in node_map.items() if v == edge[0]][0]
            child_id = [k for k, v in node_map.items() if v == edge[1]][0]
            edge_info = bom_df[(bom_df['item_id'] == child_id) & (bom_df['parent_id'] == parent_id)]

            edge_text.append(f"Parent: {parent_id}<br>"
                             f"Child: {child_id}<br>"
                             + (f"Quantity: {edge_info['quantity'].values[0]}<br>" 
                                f"Assembly Time: {edge_info['assembly_time'].values[0]} hours"
                                if not edge_info.empty else
                                "Quantity: N/A<br>Assembly Time: N/A"))

        edge_text.extend([None, None])  # Add None for gaps in line

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_colors = []
        for node in nx_g.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            item_id = [k for k, v in node_map.items() if v == node][0]
            item_info = bom_df[bom_df['item_id'] == item_id]
            if not item_info.empty:
                item_info = item_info.iloc[0]
                node_colors.append(item_info['item_type'])
            else:
                node_colors.append('Unknown')
        
        # Map node types to colors
        type_to_color = {
            'final_product': 'blue',
            'system': 'green',
            'subsystem': 'red',
            'component': 'purple',
            'subcomponent': 'orange',
            'raw_material': 'gray',
            'DUMMY_PARENT' : 'Black'
        }
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=[type_to_color.get(t, 'black') for t in node_colors],
                size=10,
                colorbar=dict(
                    thickness=0,
                
                    xanchor='left',
                    titleside='bottom'
                ),
                line_width=2
            )
        )
        
        node_adjacencies = []
        node_text = []
        for node, adjacencies in enumerate(nx_g.adjacency()):
            node_adjacencies.append(len(adjacencies[1]))
            item_id = [k for k, v in node_map.items() if v == node][0]
            item_info = bom_df[bom_df['item_id'] == item_id]
            if not item_info.empty:
                item_info = item_info.iloc[0]
                node_text.append(f"ID: {item_id}<br>"
                                f"Name: {item_info['item_name']}<br>"
                                f"Type: {item_info['item_type']}<br>"
                                f"Lead Time: {item_info['lead_time']} days<br>"
                                f"Cost: ${item_info['cost']:,}<br>"
                                f"Inventory: {item_info['inventory']}<br>"
                                f"Min Stock: {item_info['min_stock']}<br>"
                                f"Supplier: {item_info['supplier']}<br>"
                                f"Distributor: {item_info['distributor']}<br>"
                                f"Transport Method: {item_info['transport_method']}<br>"
                                f"Transport Time: {item_info['transport_time']} days<br>"
                                f"Last Updated: {item_info['timestamp']}")
            else:
                node_text.append(f"ID: {item_id}<br>"
                                f"No additional information available")

        node_trace.text = node_text
        
        # Create Plotly figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Supply Chain Network Graph',
                            titlefont_size=16,
                            showlegend=True,
                            legend=dict(x=1, y=0.5, traceorder='normal'),
                            hovermode='closest',
                            # Remove all margins and padding
                            margin=dict(l=0, r=0, t=40, b=20, pad=0),  
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), 
                            plot_bgcolor='white',
                            # Make plot auto-size to fill container
                            autosize=True 
                        ))
        
        # Add legend for node types
        for node_type, color in type_to_color.items():
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup=node_type,
                showlegend=True,
                name=node_type
            ))

        return fig
    except Exception as e:
        st.error(f"Error generating visualization: {e}")
        return None 

# Run all analyses
# bom_df = read_bom_csv('./complex_lithography_data.csv')
# g, node_map = create_supply_chain_graph(bom_df)

# # Check parent-child relationships and ensure nodes exist
# for i, row in bom_df.iterrows():
#     if pd.notna(row['parent_id']):
#         if row['parent_id'] not in node_map:
#             print(f"Parent ID {row['parent_id']} not found in node_map.")
#         if row['item_id'] not in node_map:
#             print(f"Item ID {row['item_id']} not found in node_map.")

# print(f"Number of nodes in graph: {g.number_of_nodes()}")
# print(f"Number of edges in graph: {g.number_of_edges()}")


# inventory_risk_assessment(bom_df)
# critical_component_analysis(g, node_map, bom_df)
# lead_time_analysis(bom_df)
# cost_structure_analysis(bom_df)
# supply_chain_risk_analysis(bom_df)
# visualize_graph_plotly(g, node_map, bom_df)

# print(f"Number of nodes in graph: {g.number_of_nodes()}")
# print(f"Number of edges in graph: {g.number_of_edges()}")
# print(f"Number of rows in DataFrame: {len(bom_df)}")
# print(f"Unique item_ids in DataFrame: {bom_df['item_id'].nunique()}")
# print(f"Unique parent_ids in DataFrame: {bom_df['parent_id'].nunique()}")
st.markdown(
    """
    <style>
        .reportview-container {
            max-width: 1200px; /* Adjust as needed */
            margin: 0 auto;  /* Center the content */
        }
        .stTitle h1 { 
            text-align: center;
            color: #336699;
        }
        .stSubheader h2 {
            text-align: center;
        }
        .stButton button {
            background-color: #007bff; 
            color: white;
        }
        .stMarkdown h3 { 
            margin-top: 2em; 
            color: #336699;
        }
        .stAlert {
            text-align: center;
        }
        .stDataFrame { 
            margin: 1em auto; /* Center table */
        }
        .css-18e3th9 {
            padding-top: 1em;
        }
        .css-12oz5g7 {
            width: 100%; /* Make Plotly chart responsive */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main app
st.title('BOM Based Supply Chain Graph Modeling')

# File Upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read and display dataset
    bom_df = read_bom_csv(uploaded_file)

    g,node_map = create_supply_chain_graph(bom_df=bom_df)
    # Data Summary and Insights
    st.subheader('Uploaded Dataset')
    with st.expander("Show/Hide"):  # Collapsible dataset view
        st.dataframe(bom_df)

    # Analysis in Tabs (Optional, but recommended for organization)
    tab1, tab2, tab3, tab4 ,tab5 ,tab6= st.tabs(["Visualize", "Critical Component", "Lead Time", "Costs", "Transport" , "Inventory"])

    with tab1: 
        # Show Visualization (Embedded in app)
        if st.button("Visualize"):
            fig = visualize_graph_plotly(g, node_map, bom_df)
            st.plotly_chart(fig.to_dict())
    
    with tab2:
        if st.button('Critical Component Analysis'):
            critical_component_analysis(g, node_map, bom_df)
    with tab3:    
        if st.button('Lead Time Analysis'):
            lead_time_analysis(bom_df)

    with tab4:
        if st.button('Cost Structure Analysis'):
            cost_structure_analysis(bom_df)

    with tab5:
        if st.button('Transport Analysis'):
            supply_chain_risk_analysis(bom_df)

    with tab6:
        if st.button('Inventory Risk Analysis'):
            inventory_risk_assessment(bom_df)
