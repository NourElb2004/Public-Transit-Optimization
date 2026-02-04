import time
import math
import heapq
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import folium
from folium.features import DivIcon
from streamlit_folium import folium_static
import streamlit as st
import plotly_express as px
import networkx as nx

def create_graph_manually():
    G = nx.Graph()
    
    neighborhoods = [
        (1, {'name': 'Maadi', 'population': 250000, 'type': 'Residential', 'x': 31.25, 'y': 29.96}),
        (2, {'name': 'Nasr City', 'population': 500000, 'type': 'Mixed', 'x': 31.34, 'y': 30.06}),
        (3, {'name': 'Downtown Cairo', 'population': 100000, 'type': 'Business', 'x': 31.24, 'y': 30.04}),
        (4, {'name': 'New Cairo', 'population': 300000, 'type': 'Residential', 'x': 31.47, 'y': 30.03}),
        (5, {'name': 'Heliopolis', 'population': 200000, 'type': 'Mixed', 'x': 31.32, 'y': 30.09}),
        (6, {'name': 'Zamalek', 'population': 50000, 'type': 'Residential', 'x': 31.22, 'y': 30.06}),
        (7, {'name': '6th October City', 'population': 400000, 'type': 'Mixed', 'x': 30.98, 'y': 29.93}),
        (8, {'name': 'Giza', 'population': 550000, 'type': 'Mixed', 'x': 31.21, 'y': 29.99}),
        (9, {'name': 'Mohandessin', 'population': 180000, 'type': 'Business', 'x': 31.20, 'y': 30.05}),
        (10, {'name': 'Dokki', 'population': 220000, 'type': 'Mixed', 'x': 31.21, 'y': 30.03}),
        (11, {'name': 'Shubra', 'population': 450000, 'type': 'Residential', 'x': 31.24, 'y': 30.11}),
        (12, {'name': 'Helwan', 'population': 350000, 'type': 'Industrial', 'x': 31.33, 'y': 29.85}),
        (13, {'name': 'New Administrative Capital', 'population': 50000, 'type': 'Government', 'x': 31.80, 'y': 30.02}),
        (14, {'name': 'Al Rehab', 'population': 120000, 'type': 'Residential', 'x': 31.49, 'y': 30.06}),
        (15, {'name': 'Sheikh Zayed', 'population': 150000, 'type': 'Residential', 'x': 30.94, 'y': 30.01})
    ]
    
    facilities = [
        ('F1', {'name': 'Cairo International Airport', 'type': 'Airport', 'x': 31.41, 'y': 30.11}),
        ('F2', {'name': 'Ramses Railway Station', 'type': 'Transit Hub', 'x': 31.25, 'y': 30.06}),
        ('F3', {'name': 'Cairo University', 'type': 'Education', 'x': 31.21, 'y': 30.03}),
        ('F4', {'name': 'Al-Azhar University', 'type': 'Education', 'x': 31.26, 'y': 30.05}),
        ('F5', {'name': 'Egyptian Museum', 'type': 'Tourism', 'x': 31.23, 'y': 30.05}),
        ('F6', {'name': 'Cairo International Stadium', 'type': 'Sports', 'x': 31.30, 'y': 30.07}),
        ('F7', {'name': 'Smart Village', 'type': 'Business', 'x': 30.97, 'y': 30.07}),
        ('F8', {'name': 'Cairo Festival City', 'type': 'Commercial', 'x': 31.40, 'y': 30.03}),
        ('F9', {'name': 'Qasr El Aini Hospital', 'type': 'Medical', 'x': 31.23, 'y': 30.03}),
        ('F10', {'name': 'Maadi Military Hospital', 'type': 'Medical', 'x': 31.25, 'y': 29.95})
    ]
    
    G.add_nodes_from(neighborhoods)
    G.add_nodes_from(facilities)
    
    existing_roads = [
        (1, 3, {'distance': 8.5, 'capacity': 3000, 'condition': 7, 'type': 'existing', 'cost': 0}),  
        (1, 8, {'distance': 6.2, 'capacity': 2500, 'condition': 6, 'type': 'existing', 'cost': 0}),   
        (2, 3, {'distance': 5.9, 'capacity': 2800, 'condition': 8, 'type': 'existing', 'cost': 0}),   
        (2, 5, {'distance': 4.0, 'capacity': 3200, 'condition': 9, 'type': 'existing', 'cost': 0}),   
        (3, 5, {'distance': 6.1, 'capacity': 3500, 'condition': 7, 'type': 'existing', 'cost': 0}),   
        (3, 6, {'distance': 3.2, 'capacity': 2000, 'condition': 8, 'type': 'existing', 'cost': 0}),   
        (3, 9, {'distance': 4.5, 'capacity': 2600, 'condition': 6, 'type': 'existing', 'cost': 0}),   
        (3, 10, {'distance': 3.8, 'capacity': 2400, 'condition': 7, 'type': 'existing', 'cost': 0}),  
        (4, 2, {'distance': 15.2, 'capacity': 3800, 'condition': 9, 'type': 'existing', 'cost': 0}),   
        (4, 14, {'distance': 5.3, 'capacity': 3000, 'condition': 10, 'type': 'existing', 'cost': 0}), 
        (5, 11, {'distance': 7.9, 'capacity': 3100, 'condition': 7, 'type': 'existing', 'cost': 0}),   
        (6, 9, {'distance': 2.2, 'capacity': 1800, 'condition': 8, 'type': 'existing', 'cost': 0}),    
        (7, 8, {'distance': 24.5, 'capacity': 3500, 'condition': 8, 'type': 'existing', 'cost': 0}),   
        (7, 15, {'distance': 9.8, 'capacity': 3000, 'condition': 9, 'type': 'existing', 'cost': 0}),   
        (8, 10, {'distance': 3.3, 'capacity': 2200, 'condition': 7, 'type': 'existing', 'cost': 0}),   
        (8, 12, {'distance': 14.8, 'capacity': 2600, 'condition': 5, 'type': 'existing', 'cost': 0}),  
        (9, 10, {'distance': 2.1, 'capacity': 1900, 'condition': 7, 'type': 'existing', 'cost': 0}),   
        (10, 11, {'distance': 8.7, 'capacity': 2400, 'condition': 6, 'type': 'existing', 'cost': 0}),  
        (11, 'F2', {'distance': 3.6, 'capacity': 2200, 'condition': 7, 'type': 'existing', 'cost': 0}),
        (12, 1, {'distance': 12.7, 'capacity': 2800, 'condition': 6, 'type': 'existing', 'cost': 0}),  
        (13, 4, {'distance': 45.0, 'capacity': 4000, 'condition': 10, 'type': 'existing', 'cost': 0}), 
        (14, 13, {'distance': 35.5, 'capacity': 3800, 'condition': 9, 'type': 'existing', 'cost': 0}), 
        (15, 7, {'distance': 9.8, 'capacity': 3000, 'condition': 9, 'type': 'existing', 'cost': 0}),   
        ('F1', 5, {'distance': 7.5, 'capacity': 3500, 'condition': 9, 'type': 'existing', 'cost': 0}), 
        ('F1', 2, {'distance': 9.2, 'capacity': 3200, 'condition': 8, 'type': 'existing', 'cost': 0}), 
        ('F2', 3, {'distance': 2.5, 'capacity': 2000, 'condition': 7, 'type': 'existing', 'cost': 0}), 
        ('F7', 15, {'distance': 8.3, 'capacity': 2800, 'condition': 8, 'type': 'existing', 'cost': 0}),
        ('F8', 4, {'distance': 6.1, 'capacity': 3000, 'condition': 9, 'type': 'existing', 'cost': 0})  
    ]
    
    potential_roads = [
        (1, 4, {'distance': 22.8, 'capacity': 4000, 'cost': 450, 'type': 'potential'}),
        (1, 14, {'distance': 25.3, 'capacity': 3800, 'cost': 500, 'type': 'potential'}),
        (2, 13, {'distance': 48.2, 'capacity': 4500, 'cost': 950, 'type': 'potential'}),
        (3, 13, {'distance': 56.7, 'capacity': 4500, 'cost': 1100, 'type': 'potential'}),
        (5, 4, {'distance': 16.8, 'capacity': 3500, 'cost': 320, 'type': 'potential'}),
        (6, 8, {'distance': 7.5, 'capacity': 2500, 'cost': 150, 'type': 'potential'}),
        (7, 13, {'distance': 82.3, 'capacity': 4000, 'cost': 1600, 'type': 'potential'}),
        (9, 11, {'distance': 6.9, 'capacity': 2800, 'cost': 140, 'type': 'potential'}),
        (10, 'F7', {'distance': 27.4, 'capacity': 3200, 'cost': 550, 'type': 'potential'}),
        (11, 13, {'distance': 62.1, 'capacity': 4200, 'cost': 1250, 'type': 'potential'}),
        (12, 14, {'distance': 30.5, 'capacity': 3600, 'cost': 610, 'type': 'potential'}),
        (14, 5, {'distance': 18.2, 'capacity': 3300, 'cost': 360, 'type': 'potential'}),
        (15, 9, {'distance': 22.7, 'capacity': 3000, 'cost': 450, 'type': 'potential'}),
        ('F1', 13, {'distance': 40.2, 'capacity': 4000, 'cost': 800, 'type': 'potential'}),
        ('F7', 9, {'distance': 26.8, 'capacity': 3200, 'cost': 540, 'type': 'potential'})
    ]
    
    G.add_edges_from(existing_roads)
    G.add_edges_from(potential_roads)
    return G

def prims_optimized_road_network(graph):
    """Prim's algorithm implementation with critical node priority"""
    # Create a copy of the graph to avoid modifying the original
    working_graph = graph.copy()
    
    critical_nodes = {n for n, d in working_graph.nodes(data=True) 
                     if ('medical' in str(d.get('type', '')).lower() 
                        or 'government' in str(d.get('type', '')).lower())}
    
    mst = nx.Graph()
    
    # Add all nodes from original graph to the MST
    for node, data in working_graph.nodes(data=True):
        mst.add_node(node, **data)
    
    visited = set()
    heap = []
    
    # Start with all critical nodes if they exist, otherwise arbitrary start
    start_nodes = critical_nodes if critical_nodes else {next(iter(working_graph.nodes))}
    
    for node in start_nodes:
        visited.add(node)
        for neighbor, edge_data in working_graph[node].items():
            if 'weight' in edge_data:  # Ensure weight is calculated
                heapq.heappush(heap, (edge_data['weight'], node, neighbor, edge_data))
    
    while heap and len(visited) < len(working_graph.nodes):
        weight, u, v, data = heapq.heappop(heap)
        if v in visited:
            continue
            
        # Add edge to MST with all original attributes
        mst.add_edge(u, v, **data)
        visited.add(v)
        for neighbor, edge_data in working_graph[v].items():
            if neighbor not in visited and 'weight' in edge_data:
                # Prioritize critical nodes by adjusting weight temporarily
                adj_weight = edge_data['weight']
                if neighbor in critical_nodes:
                    adj_weight *= 0.1  # Make critical nodes more attractive
                heapq.heappush(heap, (adj_weight, v, neighbor, edge_data))
    
    return mst

def analyze_prim_performance(graph):
    """Comprehensive analysis of Prim's algorithm implementation"""
    # Calculate weights first
    for u, v, d in graph.edges(data=True):
        if d['type'] == 'existing':
            condition_penalty = (11 - d['condition']) / 10
            capacity_utilization = d['capacity'] / 4500
            d['weight'] = condition_penalty / capacity_utilization
        else:
            cost = math.log10(d['cost'] + 1)
            u_pop = graph.nodes[u].get('population', 0)
            v_pop = graph.nodes[v].get('population', 0)
            pop = math.log10(u_pop + v_pop + 1000)
            
            # Check if nodes are critical
            u_critical = ('medical' in str(graph.nodes[u].get('type', '')).lower() or 
                          'government' in str(graph.nodes[u].get('type', '')).lower())
            v_critical = ('medical' in str(graph.nodes[v].get('type', '')).lower() or 
                          'government' in str(graph.nodes[v].get('type', '')).lower())
            
            priority = 3 if (u_critical or v_critical) else 1
            d['weight'] = cost / (pop * priority)
    
    # Performance metrics
    mst = prims_optimized_road_network(graph)
    
    # Solution quality metrics
    critical_nodes = {n for n, d in graph.nodes(data=True) 
                     if 'medical' in str(d.get('type', '')).lower() 
                     or 'government' in str(d.get('type', '')).lower()}
    
    # Connectivity analysis
    components = list(nx.connected_components(mst))
    critical_connected = all(any(node in comp for node in critical_nodes) 
                           for comp in components)
    
    # Edge type analysis
    edge_types = {}
    for u, v, d in mst.edges(data=True):
        edge_types[d['type']] = edge_types.get(d['type'], 0) + 1
    
    return {
        'mst': mst,
        'nodes': len(mst.nodes),
        'edges': len(mst.edges),
        'critical_nodes_connected': critical_connected,
        'connected_components': len(components),
        'edge_type_distribution': edge_types,
        'total_cost': sum(d['cost'] for _, _, d in mst.edges(data=True) 
                      if d['type'] == 'potential'),
        'graph_density': nx.density(mst)
    }

def visualize_network_on_folium(graph, mst):
    """Visualize the network on a Folium map"""
    # Find center of the map
    all_lats = [data['y'] for _, data in graph.nodes(data=True)]
    all_lons = [data['x'] for _, data in graph.nodes(data=True)]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create the map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='CartoDB positron')
    
    # Add nodes
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', '')
        population = data.get('population', 0)
        
        # Set node colors based on type
        if 'medical' in str(node_type).lower():
            color = 'red'
            radius = 10
        elif 'government' in str(node_type).lower():
            color = 'purple'
            radius = 10
        elif population > 0:  # Neighborhood
            color = 'blue'
            radius = 8
        else:  # Facility
            color = 'green'
            radius = 8
        
        # Create tooltips with node information
        tooltip_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0;">{data.get('name', node)}</h4>
            <p style="margin: 0;"><b>Type:</b> {node_type}</p>
            {f'<p style="margin: 0;"><b>Population:</b> {population:,}</p>' if population > 0 else ''}
        </div>
        """
        
        folium.CircleMarker(
            location=[data['y'], data['x']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=folium.Tooltip(tooltip_html)
        ).add_to(m)
        
        # Add node labels
        folium.map.Marker(
            [data['y'], data['x']],
            icon=DivIcon(
                icon_size=(120, 36),
                icon_anchor=(60, -10),
                html=f'<div style="font-size: 8pt; color: black; font-weight: bold; text-align: center;">{data.get("name", node)}</div>'
            )
        ).add_to(m)
    
    # Add edges
    for u, v, data in graph.edges(data=True):
        # Get node positions
        u_pos = [graph.nodes[u]['y'], graph.nodes[u]['x']]
        v_pos = [graph.nodes[v]['y'], graph.nodes[v]['x']]
        
        # Determine if edge is in MST
        in_mst = mst.has_edge(u, v) or mst.has_edge(v, u)
        
        # Set line style based on road type and whether it's in MST
        if data['type'] == 'existing':
            weight = 3 if in_mst else 1
            color = 'blue' if in_mst else 'gray'
            dash_array = '5, 5' if not in_mst else None
        else:  # potential
            weight = 3 if in_mst else 1
            color = 'red' if in_mst else 'lightgray'
            dash_array = '5, 5' if not in_mst else None
        
        # Create edge tooltip
        if data['type'] == 'potential':
            edge_tooltip = f"Distance: {data['distance']} km, Cost: {data['cost']} million EGP"
        else:
            edge_tooltip = f"Distance: {data['distance']} km"
            
        # Add line to map
        folium.PolyLine(
            locations=[u_pos, v_pos],
            weight=weight,
            color=color,
            opacity=0.8 if in_mst else 0.5,
            dash_array=dash_array,
            tooltip=edge_tooltip
        ).add_to(m)
        
        # For MST edges, add a small label with distance and cost (if potential)
        if in_mst:
            mid_point = [(u_pos[0] + v_pos[0])/2, (u_pos[1] + v_pos[1])/2]
            if data['type'] == 'potential':
                label_text = f"{data['distance']}km / ₤{data['cost']}M"
            else:
                label_text = f"{data['distance']}km"
                
            folium.map.Marker(
                mid_point,
                icon=DivIcon(
                    icon_size=(150, 20),
                    icon_anchor=(75, 10),
                    html=f'<div style="font-size: 8pt; background-color: white; border-radius: 3px; padding: 1px 3px; opacity: 0.8;">{label_text}</div>'
                )
            ).add_to(m)
    
    # Create legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <div style="margin-bottom: 5px;"><b>Node Types</b></div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: blue; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px;">Neighborhood</span>
        </div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: green; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px;">Facility</span>
        </div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: red; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px;">Medical</span>
        </div>
        <div style="margin-bottom: 5px;">
            <span style="background-color: purple; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></span>
            <span style="margin-left: 5px;">Government</span>
        </div>
        <div style="margin-bottom: 5px;"><b>Road Types</b></div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: blue; width: 20px; height: 3px; display: inline-block;"></span>
            <span style="margin-left: 5px;">Existing (MST)</span>
        </div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: red; width: 20px; height: 3px; display: inline-block;"></span>
            <span style="margin-left: 5px;">Potential (MST)</span>
        </div>
        <div style="margin-bottom: 3px;">
            <span style="background-color: gray; width: 20px; height: 3px; display: inline-block; border-style: dashed;"></span>
            <span style="margin-left: 5px;">Other Roads</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def create_mst_tables(graph, mst):
    """Create tables showing the MST edges and their properties"""
    mst_edges = []
    
    for u, v, data in mst.edges(data=True):
        u_name = mst.nodes[u].get('name', u)
        v_name = mst.nodes[v].get('name', v)
        
        if data['type'] == 'potential':
            mst_edges.append({
                'From': u_name,
                'To': v_name,
                'Distance (km)': data['distance'],
                'Cost (M EGP)': data['cost'],
                'Type': 'Potential (New)',
                'Capacity': data['capacity']
            })
        else:
            mst_edges.append({
                'From': u_name,
                'To': v_name,
                'Distance (km)': data['distance'],
                'Cost (M EGP)': '-',
                'Type': 'Existing',
                'Capacity': data['capacity'],
                'Condition': data['condition']
            })
    
    # Create DataFrame
    df = pd.DataFrame(mst_edges)
    
    # Sort by road type (potential first)
    df = df.sort_values(by=['Type'], ascending=False)
    
    return df

# Read CSV files
@st.cache_data
def load_data():
    """Loads and preprocesses data from CSV files."""
    try:
        Nodes = pd.read_csv('locations.csv')
        # Clean whitespace and ensure ID is string
        Nodes[' Name'] = Nodes[' Name'].str.strip()
        Nodes['ID'] = Nodes['ID'].astype(str) 
        Nodes[' Type'] = Nodes[' Type'].str.strip() # Clean type column

        name_to_id = pd.Series(Nodes['ID'].values, index=Nodes[' Name'].str.lower()).to_dict()
        # Ensure keys in id_to_name are strings
        id_to_name = pd.Series(Nodes[' Name'].values, index=Nodes['ID']).to_dict()

        Roads_existing = pd.read_csv('roads_existing.csv')
        # Ensure road IDs are strings
        Roads_existing['FromID'] = Roads_existing['FromID'].astype(str)
        Roads_existing['ToID'] = Roads_existing['ToID'].astype(str)
        potential_roads=pd.read_csv('roads_potential.csv')
        Facilities = pd.read_csv('facilities.csv') # Currently unused, but loaded

        Traffic_flows = pd.read_csv('traffic_flows.csv')
        # Clean RoadID if necessary, assuming format 'FromID-ToID'
        # If RoadID format is different, adjust splitting logic below

        return Nodes, Roads_existing,potential_roads, Facilities, Traffic_flows, name_to_id, id_to_name
    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Make sure 'locations.csv', 'roads_existing.csv', 'facilities.csv', and 'traffic_flows.csv' are in the correct directory.")
        st.stop() # Stop execution if essential files are missing
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        st.stop()
# Node Coordinates for Heuristic (for A*)
coordinates = {
    '1': (31.25, 29.96), '2': (31.34, 30.06), '3': (31.24, 30.04), '4': (31.47, 30.03),
    '5': (31.32, 30.09), '6': (31.22, 30.06), '7': (30.98, 29.93), '8': (31.21, 29.99),
    '9': (31.20, 30.05), '10': (31.21, 30.03), '11': (31.24, 30.11), '12': (31.33, 29.85),
    '13': (31.80, 30.02), '14': (31.49, 30.06), '15': (30.94, 30.01),
    'F1': (31.41, 30.11), 'F2': (31.25, 30.06), 'F3': (31.21, 30.03), 'F4': (31.26, 30.05),
    'F5': (31.23, 30.05), 'F6': (31.30, 30.07), 'F7': (30.97, 30.07), 'F8': (31.40, 30.03),
    'F9': (31.23, 30.03), 'F10': (31.25, 29.95)
}
@st.cache_data
def build_graph(edges_list):
    graph = defaultdict(list)
    for _, row in edges_list.iterrows():
        u, v, d = str(row['FromID']), str(row['ToID']), row['Distance']
        graph[u].append((v, d))  # Only store neighbor and distance for Dijkstra
        graph[v].append((u, d))  # Undirected graph
    return graph
@st.cache_data
def build_traffic_dict(traffic_flows_df):
    """Converts traffic flows DataFrame to a dictionary for easy lookup."""
    traffic_dict = {}
    for _, row in traffic_flows_df.iterrows():
        road_id = str(row['RoadID']) # Ensure road_id is a string
        try:
            # Assuming format 'FromID-ToID'
            from_node, to_node = road_id.split('-')
            from_node = str(from_node).strip() # Ensure node IDs are strings and stripped
            to_node = str(to_node).strip()
            
            # Handle potential non-numeric traffic values gracefully
            traffic_values = []
            for period in ['Morning', 'Afternoon', 'Evening', 'Night']:
                try:
                    traffic_values.append(float(row[period]))
                except (ValueError, TypeError):
                    traffic_values.append(0) # Default to 0 if conversion fails

            # Store both directions since the graph is undirected
            traffic_dict[(from_node, to_node)] = traffic_values
            traffic_dict[(to_node, from_node)] = traffic_values
        except ValueError:
            st.warning(f"Skipping invalid RoadID format: {road_id}")
        except KeyError as e:
             st.warning(f"Missing traffic column {e} for RoadID {road_id}. Using default 0.")
             # Fill missing columns with 0 if needed, though load_data should handle this
             traffic_values = [0, 0, 0, 0]
             traffic_dict[(from_node, to_node)] = traffic_values
             traffic_dict[(to_node, from_node)] = traffic_values

    return traffic_dict

def calculate_travel_time(distance, node, neighbor, traffic_flows, time_period, is_priority=False):
    """Calculate travel time based on distance and traffic conditions"""
    # Default speed without traffic (km/h)
    default_speed = 60
    
    # Maximum road capacity (vehicles/hour)
    max_capacity = 4000
    
    # Time period index
    time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[time_period]
    
    # Check if we have traffic data for this road segment
    edge = (node, neighbor)
    
    if edge in traffic_flows:
        # Get traffic flow for current time period
        flow = traffic_flows[edge][time_idx]
        
        # Calculate speed based on traffic flow
        # As traffic approaches capacity, speed decreases
        if is_priority:
            # Higher base speed and better congestion handling for emergency/A* routes
            default_speed *= 1.5  # 50% higher base speed
            traffic_factor = min(flow / (max_capacity * 1.5), 0.7)  # Less impact from traffic
        else:
            traffic_factor = min(flow / max_capacity, 0.9)  # Cap at 90% reduction
            
        adjusted_speed = default_speed * (1 - traffic_factor)
        
        # Minimum speed of 10 km/h even in heavy traffic (20 km/h for priority routes)
        adjusted_speed = max(adjusted_speed, 20 if is_priority else 10)
        
        # Calculate travel time (hours)
        travel_time = distance / adjusted_speed
    else:
        # No traffic data available, use default speed
        travel_time = distance / (default_speed * 1.5 if is_priority else default_speed)
        
    return travel_time * 60  # Convert to minutes

def heuristic(a, b):
    """Euclidean distance heuristic for A*"""
    ax, ay = coordinates.get(a, (0, 0))
    bx, by = coordinates.get(b, (0, 0))
    return math.sqrt((ax - bx)**2 + (ay - by)**2)


def dijkstra_time_dependent(graph, traffic_flows, start_id, goal_id, time_period):
    """Dijkstra algorithm that prioritizes routes with lower congestion"""
    pq = [(0, 0, start_id)]  # (total_time, congestion_score, node)
    visited = set()
    travel_times = {start_id: 0}  # Store travel time in minutes
    congestion_scores = {start_id: 0}  # Track cumulative congestion score
    prev = {}

    # Time period index
    time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[time_period]
    
    # Maximum road capacity (vehicles/hour)
    max_capacity = 4000

    while pq:
        _, current_congestion_score, node = heapq.heappop(pq)
        
        if node == goal_id:
            break
            
        if node in visited:
            continue
            
        visited.add(node)
        
        for neighbor, distance in graph.get(node, []):
            if neighbor not in visited:
                # Calculate travel time based on traffic
                edge = (node, neighbor)
                
                # Get traffic flow for current time period
                flow = traffic_flows.get(edge, [0, 0, 0, 0])[time_idx]
                
                # Calculate congestion level (0-1 scale)
                congestion_level = min(flow / max_capacity, 1.0)
                
                # Calculate travel time
                segment_time = calculate_travel_time(distance, node, neighbor, traffic_flows, time_period, is_priority=False)
                
                # Calculate new cumulative congestion score (weighted by segment distance)
                new_congestion_score = current_congestion_score + (congestion_level * distance)
                
                new_time = travel_times[node] + segment_time
                
                if (neighbor not in travel_times) or (new_time < travel_times[neighbor]) or \
                   (new_time == travel_times[neighbor] and new_congestion_score < congestion_scores.get(neighbor, float('inf'))):
                    travel_times[neighbor] = new_time
                    congestion_scores[neighbor] = new_congestion_score
                    prev[neighbor] = node
                    
                    # Use combined metric that considers both time and congestion
                    # Higher weight on congestion to prefer less congested routes
                    combined_metric = new_time * (1 + 0.5 * congestion_level)
                    
                    heapq.heappush(pq, (combined_metric, new_congestion_score, neighbor))

    # Reconstruct path
    path = []
    curr = goal_id
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    
    if path:
        path.append(start_id)
        path.reverse()
        
    return path, travel_times.get(goal_id, float('inf'))

def a_star_time_dependent(graph, traffic_flows, start, goal, time_period):
    """A* algorithm that prioritizes routes with lower congestion"""
    # Initial heuristic estimate (in minutes, assuming average speed of 40 km/h)
    h_estimate = heuristic(start, goal) * 60 / 40
    
    # Time period index
    time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[time_period]
    
    # Maximum road capacity (vehicles/hour)
    max_capacity = 4000
    
    pq = [(h_estimate, 0, 0, start)]  # (estimated_total, time_so_far, congestion_score, node)
    visited = set()
    travel_times = {start: 0}  # Store travel time in minutes
    congestion_scores = {start: 0}  # Track cumulative congestion score
    prev = {}

    while pq:
        _, time_so_far, current_congestion_score, node = heapq.heappop(pq)
        
        if node == goal:
            break
            
        if node in visited:
            continue
            
        visited.add(node)
        
        for neighbor, distance in graph.get(node, []):
            if neighbor not in visited:
                # Calculate travel time based on traffic
                edge = (node, neighbor)
                
                # Get traffic flow for current time period
                flow = traffic_flows.get(edge, [0, 0, 0, 0])[time_idx]
                
                # Calculate congestion level (0-1 scale)
                congestion_level = min(flow / max_capacity, 1.0)
                
                # Calculate travel time
                segment_time = calculate_travel_time(distance, node, neighbor, traffic_flows, time_period, is_priority=True)
                
                # Calculate new cumulative congestion score (weighted by segment distance)
                new_congestion_score = current_congestion_score + (congestion_level * distance)
                
                new_time = time_so_far + segment_time
                
                if (neighbor not in travel_times) or (new_time < travel_times[neighbor]) or \
                   (new_time == travel_times[neighbor] and new_congestion_score < congestion_scores.get(neighbor, float('inf'))):
                    travel_times[neighbor] = new_time
                    congestion_scores[neighbor] = new_congestion_score
                    prev[neighbor] = node
                    
                    # Calculate heuristic (estimate of remaining time)
                    h = heuristic(neighbor, goal) * 60 / 40  # Convert to minutes using avg speed
                    
                    # Use combined metric that considers both time and congestion
                    # Higher weight on congestion to prefer less congested routes
                    combined_metric = new_time * (1 + 0.5 * congestion_level) + h
                    
                    heapq.heappush(pq, (combined_metric, new_time, new_congestion_score, neighbor))

    # Reconstruct path
    path = []
    curr = goal
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    
    if path:
        path.append(start)
        path.reverse()
        
    return path, travel_times.get(goal, float('inf'))

def dijkstra_congestion_dependent(graph, traffic_flows, start_id, goal_id, time_period):
    """Dijkstra algorithm that strongly prioritizes routes with lower congestion"""
    # Time period index for traffic flow lookup
    time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[time_period]
    
    # Maximum road capacity (vehicles/hour)
    max_capacity = 4000
    
    # Priority queue: (combined_metric, time_so_far, congestion_sum, node)
    pq = [(0, 0, 0, start_id)]
    visited = set()
    
    # Track metrics for each node
    best_metrics = {
        start_id: {
            'time': 0,
            'congestion_sum': 0,
            'avg_congestion': 0,
            'distance': 0
        }
    }
    prev = {}

    while pq:
        _, time_so_far, congestion_sum, node = heapq.heappop(pq)
        
        if node == goal_id:
            break
            
        if node in visited:
            continue
            
        visited.add(node)
        
        for neighbor, distance in graph.get(node, []):
            if neighbor in visited:
                continue
                
            # Get traffic flow for this segment
            edge = (node, neighbor)
            flow = traffic_flows.get(edge, [0, 0, 0, 0])[time_idx]
            
            # Calculate congestion level (0-1 scale)
            congestion_level = min(flow / max_capacity, 1.0)
            
            # Calculate travel time for this segment
            segment_time = calculate_travel_time(distance, node, neighbor, traffic_flows, time_period)
            
            # Update metrics
            new_time = time_so_far + segment_time
            new_congestion_sum = congestion_sum + flow  # Raw flow sum
            new_distance = best_metrics[node]['distance'] + distance
            
            # Calculate average congestion (weighted by distance)
            new_avg_congestion = new_congestion_sum / new_distance if new_distance > 0 else 0
            
            # Check if this is a better path to the neighbor
            is_better_path = False
            if neighbor not in best_metrics:
                is_better_path = True
            else:
                # STRONGLY prefer paths with lower average congestion
                # Only choose a more congested path if it saves significant time
                current_avg_congestion = best_metrics[neighbor]['avg_congestion']
                current_time = best_metrics[neighbor]['time']
                
                # If new path has at least 15% less congestion, prefer it
                if new_avg_congestion < current_avg_congestion * 0.85:
                    is_better_path = True
                # If congestion is similar (within 15%), choose faster path
                elif new_avg_congestion < current_avg_congestion * 1.15 and new_time < current_time:
                    is_better_path = True
                # If new path is much faster (30% or more), choose it despite congestion
                elif new_time < current_time * 0.7:
                    is_better_path = True
            
            if is_better_path:
                best_metrics[neighbor] = {
                    'time': new_time,
                    'congestion_sum': new_congestion_sum,
                    'avg_congestion': new_avg_congestion,
                    'distance': new_distance
                }
                prev[neighbor] = node
                
                # Combined metric strongly favors lower congestion
                # The 3.0 multiplier gives congestion much higher weight than time
                combined_metric = new_time + (3.0 * new_avg_congestion * new_time)
                
                heapq.heappush(pq, (combined_metric, new_time, new_congestion_sum, neighbor))

    # Reconstruct path
    path = []
    curr = goal_id
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    
    if path:
        path.append(start_id)
        path.reverse()
        
    return path, best_metrics.get(goal_id, {}).get('time', float('inf'))

def a_star_congestion_dependent(graph, traffic_flows, start, goal, time_period):
    """A* algorithm that strongly prioritizes routes with lower congestion"""
    # Time period index for traffic flow lookup
    time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[time_period]
    
    # Maximum road capacity (vehicles/hour)
    max_capacity = 4000
    
    # Initial heuristic estimate (in minutes, assuming average speed of 40 km/h)
    h_estimate = heuristic(start, goal) * 60 / 40
    
    # Priority queue: (combined_metric, time_so_far, congestion_sum, node)
    pq = [(h_estimate, 0, 0, start)]
    visited = set()
    
    # Track metrics for each node
    best_metrics = {
        start: {
            'time': 0,
            'congestion_sum': 0,
            'avg_congestion': 0,
            'distance': 0
        }
    }
    prev = {}

    while pq:
        _, time_so_far, congestion_sum, node = heapq.heappop(pq)
        
        if node == goal:
            break
            
        if node in visited:
            continue
            
        visited.add(node)
        
        for neighbor, distance in graph.get(node, []):
            if neighbor in visited:
                continue
                
            # Get traffic flow for this segment
            edge = (node, neighbor)
            flow = traffic_flows.get(edge, [0, 0, 0, 0])[time_idx]
            
            # Calculate congestion level (0-1 scale)
            congestion_level = min(flow / max_capacity, 1.0)
            
            # Calculate travel time for this segment
            segment_time = calculate_travel_time(distance, node, neighbor, traffic_flows, time_period)
            
            # Update metrics
            new_time = time_so_far + segment_time
            new_congestion_sum = congestion_sum + flow  # Raw flow sum
            new_distance = best_metrics[node]['distance'] + distance
            
            # Calculate average congestion (weighted by distance)
            new_avg_congestion = new_congestion_sum / new_distance if new_distance > 0 else 0
            
            # Check if this is a better path to the neighbor
            is_better_path = False
            if neighbor not in best_metrics:
                is_better_path = True
            else:
                # STRONGLY prefer paths with lower average congestion
                # Only choose a more congested path if it saves significant time
                current_avg_congestion = best_metrics[neighbor]['avg_congestion']
                current_time = best_metrics[neighbor]['time']
                
                # If new path has at least 15% less congestion, prefer it
                if new_avg_congestion < current_avg_congestion * 0.85:
                    is_better_path = True
                # If congestion is similar (within 15%), choose faster path
                elif new_avg_congestion < current_avg_congestion * 1.15 and new_time < current_time:
                    is_better_path = True
                # If new path is much faster (30% or more), choose it despite congestion
                elif new_time < current_time * 0.7:
                    is_better_path = True
            
            if is_better_path:
                best_metrics[neighbor] = {
                    'time': new_time,
                    'congestion_sum': new_congestion_sum,
                    'avg_congestion': new_avg_congestion,
                    'distance': new_distance
                }
                prev[neighbor] = node
                
                # Calculate heuristic (estimate of remaining time)
                h = heuristic(neighbor, goal) * 60 / 40  # Convert to minutes using avg speed
                
                # Combined metric strongly favors lower congestion
                # The 3.0 multiplier gives congestion much higher weight than time
                combined_metric = (new_time + h) + (3.0 * new_avg_congestion * new_time)
                
                heapq.heappush(pq, (combined_metric, new_time, new_congestion_sum, neighbor))

    # Reconstruct path
    path = []
    curr = goal
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    
    if path:
        path.append(start)
        path.reverse()
        
    return path, best_metrics.get(goal, {}).get('time', float('inf'))
def plot_folium_path(path, nodes_df, id_to_name_map):
    """Visualize the path on a Folium map."""
    if not path or len(path) < 2:
        # st.warning("Path is empty or too short to display.") # Reduce warnings
        return None

    # Create map centered roughly on Cairo
    m = folium.Map(location=[30.0444, 31.2357], zoom_start=11) # Adjusted zoom
    path_coords = []
    missing_coords = []

    # Ensure the id_to_name map uses string keys consistent with path IDs
    id_to_name = {str(k): v for k, v in id_to_name_map.items()}

    # Ensure Nodes DataFrame uses string IDs for lookup
    try:
         nodes_df_str_id = nodes_df.copy()
         nodes_df_str_id['ID'] = nodes_df_str_id['ID'].astype(str)
         nodes_lookup = nodes_df_str_id.set_index('ID')
    except Exception as e:
         st.error(f"Failed to prepare nodes DataFrame for coordinate lookup: {e}")
         return None # Cannot proceed without node lookup


    for node_id in path:
        node_id_str = str(node_id) # Ensure string ID
        lat, lon = None, None

        # Try getting from Nodes DataFrame first (assuming 'Y-coordinate', 'X-coordinate')
        if node_id_str in nodes_lookup.index:
             node_data = nodes_lookup.loc[node_id_str]
             # Handle potential Series if multiple nodes share an ID (shouldn't happen with unique IDs)
             if isinstance(node_data, pd.DataFrame):
                 node_data = node_data.iloc[0] # Take the first one

             try:
                 # Check if columns exist and are not NaN
                 if 'Y-coordinate' in node_data.index and 'X-coordinate' in node_data.index and \
                    pd.notna(node_data['Y-coordinate']) and pd.notna(node_data['X-coordinate']):
                      lat = float(node_data['Y-coordinate']) # Latitude
                      lon = float(node_data['X-coordinate']) # Longitude
                 else:
                      # st.warning(f"Coordinate columns missing or NaN for node {node_id_str} in locations.csv")
                      pass # Will fallback to hardcoded coordinates
             except (ValueError, TypeError) as e:
                 # st.warning(f"Error parsing coordinates for node {node_id_str} from locations.csv: {e}")
                 pass # Will fallback
             except AttributeError: # Handle cases where node_data might not be a Series/DataFrame row
                  pass


        # Fallback to hardcoded coordinates if not found or invalid in DataFrame
        if lat is None or lon is None:
            if node_id_str in coordinates:
                # Hardcoded coordinates format: (Longitude, Latitude) based on earlier analysis
                lon_coord, lat_coord = coordinates[node_id_str]
                lat = lat_coord
                lon = lon_coord
            else:
                missing_coords.append(node_id_str)
                continue # Skip node if coordinates are missing everywhere

        # Append valid coordinates (Latitude, Longitude for Folium)
        # Add basic validation for coordinate range
        if -90 <= lat <= 90 and -180 <= lon <= 180:
             path_coords.append([lat, lon])
        else:
             # st.warning(f"Invalid coordinates ({lat}, {lon}) for node {node_id_str}. Skipping.")
             missing_coords.append(node_id_str)


    if missing_coords:
         # st.warning(f"Missing or invalid coordinates for nodes: {', '.join(list(set(missing_coords)))}. They won't be shown on the map.")
         pass # Reduce warnings in UI

    if len(path_coords) < 2:
        # st.error("Not enough valid coordinates to draw a path.") # Reduce warnings
        return None

    # Draw path
    try:
        folium.PolyLine(
            path_coords,
            color='blue', # Consistent color
            weight=5,
            opacity=0.8,
            tooltip="Calculated Route"
        ).add_to(m)
    except Exception as e:
        st.error(f"Error adding PolyLine to map: {e}")
        return None # Return None if PolyLine fails

    # Add markers for start, end, and intermediate points
    # Create a mapping from node ID to its index in the original path
    path_indices = {str(node_id): i for i, node_id in enumerate(path)}
    # Keep track of added markers to avoid duplicates if coords were missing/reused
    added_markers = set()

    for coord_idx, (lat, lon) in enumerate(path_coords):
         # Find which node_id this coordinate belongs to
         # This requires finding which original path node corresponds to this valid coordinate
         # We need to iterate through the original path and find the first node
         # whose coordinates match lat, lon and hasn't been marked yet.
         
         found_node_id = None
         for node_id_in_path in path:
              node_id_str_in_path = str(node_id_in_path)
              if node_id_str_in_path in missing_coords or node_id_str_in_path in added_markers:
                   continue # Skip nodes with bad coords or already marked

              # Check if this node's coordinate matches the current [lat, lon]
              node_lat, node_lon = None, None
              # Re-fetch coordinates for comparison (could be optimized)
              if node_id_str_in_path in nodes_lookup.index:
                   node_data = nodes_lookup.loc[node_id_str_in_path]
                   if isinstance(node_data, pd.DataFrame): node_data = node_data.iloc[0]
                   try:
                        if 'Y-coordinate' in node_data.index and 'X-coordinate' in node_data.index and pd.notna(node_data['Y-coordinate']) and pd.notna(node_data['X-coordinate']):
                            node_lat = float(node_data['Y-coordinate'])
                            node_lon = float(node_data['X-coordinate'])
                   except (ValueError, TypeError, AttributeError): pass
              if node_lat is None or node_lon is None:
                  if node_id_str_in_path in coordinates:
                      lon_coord_hc, lat_coord_hc = coordinates[node_id_str_in_path]
                      node_lat, node_lon = lat_coord_hc, lon_coord_hc
              
              # Compare coordinates with tolerance
              if node_lat is not None and node_lon is not None and \
                 abs(node_lat - lat) < 1e-6 and abs(node_lon - lon) < 1e-6:
                  found_node_id = node_id_str_in_path
                  break # Found the node for this coordinate

         if found_node_id:
             i = path_indices[found_node_id] # Get original index in path
             node_name = id_to_name.get(found_node_id, f"Unknown (ID: {found_node_id})")
             
             is_start = (i == 0)
             is_end = (i == len(path) - 1)

             if is_start: # Start node
                 icon_color = 'green'
                 icon_type = 'play'
                 popup_text = f"Start: {node_name}"
             elif is_end: # End node
                 icon_color = 'red'
                 icon_type = 'stop'
                 popup_text = f"End: {node_name}"
             else: # Intermediate node
                 icon_color = 'lightblue'
                 icon_type = 'info-sign'
                 popup_text = f"{i}: {node_name}"

             try:
                  folium.Marker(
                      [lat, lon],
                      popup=popup_text,
                      tooltip=node_name,
                      icon=folium.Icon(color=icon_color, icon=icon_type, prefix='glyphicon')
                  ).add_to(m)
                  added_markers.add(found_node_id) # Mark as added
             except Exception as e:
                  st.warning(f"Error adding marker for {node_name}: {e}")


    # Auto-zoom map to fit the path
    try:
        # Check if path_coords actually contains valid bounds
        if path_coords and len(path_coords) > 0:
             # Calculate bounds manually for more control if fit_bounds fails
             min_lat = min(p[0] for p in path_coords)
             max_lat = max(p[0] for p in path_coords)
             min_lon = min(p[1] for p in path_coords)
             max_lon = max(p[1] for p in path_coords)
             
             # Add padding only if the bounds are not identical
             if abs(max_lat - min_lat) < 1e-6 and abs(max_lon - min_lon) < 1e-6:
                  # Single point, add small padding
                  padding = 0.01
                  bounds = [[min_lat - padding, min_lon - padding], [max_lat + padding, max_lon + padding]]
             else:
                  bounds = [[min_lat, min_lon], [max_lat, max_lon]]

             m.fit_bounds(bounds, padding=(0.01, 0.01)) # Use degree padding
        # else: map remains at default zoom centered on Cairo
    except Exception as e:
        st.warning(f"Could not auto-zoom map: {e}. Map may not be centered correctly.")

    return m
def get_location_name(node_id, id_to_name):
    """Helper function to correctly get location name from ID with proper type handling"""
    # Try different ways to look up the name
    if node_id in id_to_name:
        return id_to_name[node_id]
    
    # Try with int conversion (if node_id is a string but id_to_name has int keys)
    try:
        int_id = int(node_id)
        if int_id in id_to_name:
            return id_to_name[int_id]
    except (ValueError, TypeError):
        pass
    
    # Finally, use string version
    str_id = str(node_id)
    if str_id in id_to_name:
        return id_to_name[str_id]
    
    return f"Unknown (ID: {node_id})"

def get_path_details(path, graph, traffic_flows, time_period, id_to_name_map, is_priority=False):
    """Get detailed information about each segment in the path."""
    if not path: return [], 0, 0, 0 # Return 4 values now

    segments = []
    total_distance = 0
    total_time = 0
    total_weighted_congestion = 0 # To store the sum of (flow * distance) for the path

    time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}.get(time_period, 0)

    # Ensure keys in the map are strings
    id_to_name = {str(k): v for k, v in id_to_name_map.items()}

    for i in range(len(path) - 1):
        from_node = str(path[i])
        to_node = str(path[i+1])

        # Find the distance from the graph (which uses string IDs)
        distance = next((w for n, w in graph.get(from_node, []) if str(n) == to_node), None)

        if distance is None or not isinstance(distance, (int, float)) or distance < 0:
            continue # Skip segment if distance not found or invalid

        # Get traffic flow for this segment
        edge = (from_node, to_node)
        traffic = 0
        if edge in traffic_flows:
             try:
                 traffic = traffic_flows[edge][time_idx]
                 if not isinstance(traffic, (int, float)): traffic = 0 # Ensure numeric
             except (IndexError, TypeError):
                 traffic = 0 # Default on error
        if traffic < 0: traffic = 0

        # Calculate travel time
        travel_time = calculate_travel_time(distance, from_node, to_node, traffic_flows, time_period, is_priority)

        if travel_time == float('inf'):
            continue

        from_name = id_to_name.get(from_node, f"Unknown (ID: {from_node})")
        to_name = id_to_name.get(to_node, f"Unknown (ID: {to_node})")

        segments.append({
            "From": from_name,
            "To": to_name,
            "Distance (km)": f"{distance:.2f}",
            f"Traffic ({time_period.capitalize()})": f"{traffic:.0f} veh/h", # Use integer format for traffic
            "Time (min)": f"{travel_time:.2f}"
        })

        total_distance += distance
        total_time += travel_time
        total_weighted_congestion += traffic * distance # Accumulate congestion score

    # Ensure returning 4 values consistently
    return segments, total_distance, total_time, total_weighted_congestion

def run_gui():
    st.set_page_config(
    page_title="Cairo Transportation System",
    page_icon="🌐",
    layout="wide"
)
    st.sidebar.image("Logo.png")
   # st.sidebar.title("📍 Cairo Transportation System")
    Nodes, Roads_existing,potential_roads, Facilities, Traffic_flows, name_to_id, id_to_name = load_data()
    graph = build_graph(Roads_existing)
    traffic_flows_dict = build_traffic_dict(Traffic_flows)

    # Time period selection
    st.sidebar.header("Settings")
    time_period = st.sidebar.radio(
        "Time of Day",
        ["morning", "afternoon", "evening", "night"],
        index=0, # Default to morning
        key='time_period_radio',
        help="Select the time of day to account for different traffic patterns."
    )
    locations = sorted([str(name) for name in Nodes[' Name'].unique()])
    if 'start_location' not in st.session_state:
        st.session_state.start_location = locations[0] if locations else None
    if 'goal_location' not in st.session_state:
        st.session_state.goal_location = locations[1] if len(locations) > 1 else None

    start = st.sidebar.selectbox(
        "Starting Location", 
        locations, 
        index=locations.index(st.session_state.start_location) if st.session_state.start_location in locations else 0,
        key='start_loc_select'
        )
    goal = st.sidebar.selectbox(
        "Destination", 
        locations, 
        index=locations.index(st.session_state.goal_location) if st.session_state.goal_location in locations else (1 if len(locations) > 1 else 0),
        key='goal_loc_select'
        )
        
    # Update session state
    st.session_state.start_location = start
    st.session_state.goal_location = goal


    # Get corresponding IDs (ensure lowercase for lookup and result is string)
    start_id = name_to_id.get(start.lower()) if start else None
    goal_id = name_to_id.get(goal.lower()) if goal else None

    if not start_id or not goal_id:
         st.sidebar.error("Please select valid start and end locations.")
         st.stop() # Stop if locations are invalid

    start_id, goal_id = str(start_id), str(goal_id) # Ensure they are strings for algorithms


    # Check if start or goal is a 'Government' type location
    # Ensure Nodes index is set to ID (string) for efficient lookup
    nodes_indexed = Nodes.set_index('ID')
    try:
        start_is_gov = nodes_indexed.loc[start_id, ' Type'].strip().lower() == 'government'
    except KeyError:
        start_is_gov = False # Node ID not found or Type column missing
    except AttributeError:
        start_is_gov = False # Type is likely not a string (e.g., NaN)
        
    try:
        goal_is_gov = nodes_indexed.loc[goal_id, ' Type'].strip().lower() == 'government'
    except KeyError:
        goal_is_gov = False
    except AttributeError:
        goal_is_gov = False
        
    is_priority_route = start_is_gov or goal_is_gov

    # Reorganized tabs structure as requested
    tab1, tab2, tab3= st.tabs(["🕸 Infrastructure Network", "⬆⬇ Path Finder","🚇 Public Transportaion"])
    
    # First tab - Infrastructure Network (formerly tab4)
    with tab1:
        G = create_graph_manually()
        analysis = analyze_prim_performance(G)
        mst = analysis['mst']
       # st.title("Cairo Road Network Optimization")
       # st.markdown("This application visualizes the Cairo road network and applies Prim's algorithm to find the optimal network configuration.")
        network_tabs=st.tabs(["🧠 Prim's Algorithm Optimization","🗺 Network Visualization","🧮 Calculation Methodology"])
        with network_tabs[0]:
            st.title("Prim's Algorithm Optimization")
            if st.button("Optimize Network"):
                # Run analysis
                with st.spinner("Running Prim's algorithm and analyzing the network..."):
                
                
                # Display analysis results

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Network Statistics")
                        st.write(f"Nodes in MST: {analysis['nodes']}")
                        st.write(f"Edges in MST: {analysis['edges']}")
                       # st.write(f"Critical Nodes Connected: {'Yes' if analysis['critical_nodes_connected'] else 'No'}")
                        st.write(f"Total New Road Cost: {analysis['total_cost']} million EGP")
                        st.write(f"Graph Density: {analysis['graph_density']:.4f}")
                    with col2:
                        st.markdown("### Edge Type Distribution")
                        edge_types = analysis['edge_type_distribution']
                        # Create a pie chart for edge types
                        labels = [f"{key.capitalize()} ({value})" for key, value in edge_types.items()]
                        values = list(edge_types.values())
                        fig = px.pie(
                        names=labels,
                        values=values,
                        color_discrete_sequence=px.colors.sequential.RdBu
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        # Display the chart in Streamlit
                        st.plotly_chart(fig, use_container_width=True)
                with st.expander("Original Graph Details"):
                    st.write(f"Total nodes: {len(G.nodes())}")
                    st.write(f"Neighborhoods: {sum(1 for n,d in G.nodes(data=True) if 'population' in d)}")
                    st.write(f"Facilities: {sum(1 for n,d in G.nodes(data=True) if 'population' not in d)}")
                    st.write(f"Existing roads: {sum(1 for u,v,d in G.edges(data=True) if d['type'] == 'existing')}")
                    st.write(f"Potential roads: {sum(1 for u,v,d in G.edges(data=True) if d['type'] == 'potential')}")
        with network_tabs[1]:
            # Display map
            st.title("Network Visualization")
            if st.button("Show Map"):
                st.markdown("The map shows the Cairo road network with the Minimum Spanning Tree (MST) highlighted. "
                            "Blue lines represent existing roads in the MST, red lines represent potential new roads in the MST, "
                            "and gray dashed lines show roads not in the MST.")

                map_col1, map_col2 = st.columns([3, 1])

                with map_col1:
                    m = visualize_network_on_folium(G, mst)
                    folium_static(m, width=800, height=600)

                with map_col2:
                    st.markdown("### Legend")
                    st.markdown("**Node Types:**")
                    st.markdown("🔵 Neighborhood")
                    st.markdown("🟢 Facility")
                    st.markdown("🔴 Medical")
                    st.markdown("🟣 Government")

                    st.markdown("**Road Types:**")
                    st.markdown("🔵 Existing (MST)")
                    st.markdown("🔴 Potential (MST)")
                    st.markdown("⚫ Other Roads (Not in MST)")

                # Display MST edge table
                exp_col1,exp_col2=st.columns(2)
                mst_table = create_mst_tables(G, mst)
                potential_roads = mst_table[mst_table['Type'] == 'Potential (New)']
                with exp_col1:
                    with st.expander("Summary of New Roads"):
                    
                        if not potential_roads.empty:
                            total_cost = potential_roads['Cost (M EGP)'].sum()
                            total_distance = potential_roads['Distance (km)'].sum()
                            st.markdown(f"**Total New Roads:** {len(potential_roads)}")
                            st.markdown(f"**Total Distance:** {total_distance:.1f} km")
                            st.markdown(f"**Total Cost:** {total_cost:.1f} million EGP")
                        else:
                            st.markdown("No new roads are required in the optimized network.")
                    
                with exp_col2:
                    with st.expander("New Roads to Be Built"):
                        st.dataframe(potential_roads[['From', 'To', 'Distance (km)', 'Cost (M EGP)', 'Capacity']], 
                                use_container_width=True)
                with st.expander("MST Road Connections"):
                        st.markdown("This table shows all roads included in the Minimum Spanning Tree:")

                        st.dataframe(mst_table, use_container_width=True)
                

        with network_tabs[2]:
            # Add details about calculation methodology
            st.title("🚀 Optimization Calculation Methodology")

            st.markdown("""
            This application uses **Prim's Algorithm** to optimize road connections based on a weighted graph.\n
            The methodology varies depending on the type of road:
                        
            ---
            ### 🛣️ Existing Roads
            **Weight Formula:**
            (11 - condition) / 10 ÷ (capacity / 4500)
            - ✅ Roads in **better condition** (higher condition score) are prioritized.
            - 📦 Roads with **higher capacity** are favored to ensure better flow.
            ---
            ### 🏗️ Potential Roads
            **Weight Formula:**
            log₁₀(cost + 1) ÷ [log₁₀(population₁ + population₂ + 1000) × priority]
            - 💰 Lower **construction cost** leads to better optimization.
            - 👥 Roads connecting **more populated areas** are prioritized.
            - 🏥🏛️ Roads leading to **medical or government facilities** get a **3× priority boost**.
            ---
            🔍 This approach ensures that both cost-efficiency and social importance are considered in the road network optimization process.
            """)

        # Second tab - Path Finder with subtabs
    with tab2:
        path_finder_tabs = st.tabs(["🚦 Time-Optimized Path Finder", "📈 Strategies", "📊 Analysis & Comparisons"])
        
        # Subtab 1 - Time-Optimized Path Finder (formerly tab1)
        with path_finder_tabs[0]:
            st.title("Time-Optimized Path Finder")
            traffic_flows = build_traffic_dict(Traffic_flows)

            # Initialize path and time
            emergency_vehicle=st.checkbox("Emergency Vehicle Route")
            # Calculate path on demand or if inputs change (using a button is clearer)
            if st.button("Find Fastest Path", key="find_time_path"):
                if start_id == goal_id:
                    st.warning("Start and destination locations are the same.")
                if is_priority_route or emergency_vehicle:
                    algo_name = "A* (Time-Optimized)"
                    path_func = a_star_time_dependent
                    st.info("Using A* search Algorithm.")
                else:
                    algo_name = "Dijkstra (Time-Optimized)"
                    path_func = dijkstra_time_dependent
                    st.info("Using Dijkstra search Algorithm.")
                try:
                    with st.spinner(f'Calculating fastest path using {algo_name}...'):
                        import time as timing_module  # Local import to avoid naming conflicts
                        start_time_calc = timing_module.time()
                        path, travel_time = path_func(graph, traffic_flows_dict, start_id, goal_id, time_period)
                        calc_duration = timing_module.time() - start_time_calc
                    if not path or travel_time == float('inf'):
                        st.error(f"❌ No path exists between {start} and {goal} with current settings!")
                    else:
                        st.success(f"✅ Fastest path found in {calc_duration:.2f} seconds!")
                        st.subheader("Results")
                        segments, total_distance, actual_total_time, CONGESTION_SCORE = get_path_details(
                            path, graph, traffic_flows_dict, time_period, id_to_name, 
                            is_priority=(is_priority_route or emergency_vehicle)  # Pass is_priority based on route type
                        )
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Est. Travel Time", f"{actual_total_time:.1f} min")
                        with col2:
                            st.metric("Total Distance", f"{total_distance:.2f} km")
                        with col3:
                            st.metric("Number of Stops", f"{len(path)} ({len(path)-1} segments)")
                    st.subheader("Route Map")
                    map_object = plot_folium_path(path, Nodes, id_to_name)  
                    if map_object:
                        folium_static(map_object, width=700, height=500) # Adjust size as needed
                    else:
                        st.warning("Could not display map.")
                    with st.expander("Show Route Steps and Details"):
                             st.write("#### Route Overview:")
                             route_names = [get_location_name(node, id_to_name) for node in path]
                             st.write(" → ".join(route_names))
                             
                             st.write("#### Segment Details:")
                             if segments:
                                 st.dataframe(pd.DataFrame(segments).set_index('From'))
                             else:
                                 st.write("No segments to display.")
                except Exception as e:
                    st.error(f"Error calculating path: {str(e)}")
                    st.exception(e)
        
        # Subtab 2 - Strategies (formerly tab2)
        with path_finder_tabs[1]:
            st.title("Routing Strategies")

            st.markdown("""
            ## 🧭 Path Planning Approaches

            This application provides two routing modes: one that optimizes for **travel time** and one that represents a **default (unoptimized)** route for baseline comparison.
            """)

            # Create columns for the two strategies
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("⏱️ Time-Optimized Path")
                st.markdown("""
                - **Goal**: Minimize total travel time
                - **Algorithm**: Dijkstra's or A* algorithm
                - **Ideal for**:
                  - Emergency responders
                  - Time-sensitive deliveries
                  - medical routes
                - **Key Features**:
                  - Takes live or estimated traffic into account
                  - May use slightly longer roads if they're faster overall
                  - Prioritizes roads with higher capacity and better conditions
                """)

                # Example calculation
                with st.expander("📊 Technical Example"):
                    st.markdown("""
                    ```python
                    def dijkstra_time_optimized(graph, traffic_data, start, end, time_period):
                        pq = [(0, 0, start)]  # (total_time, congestion_penalty, node)
                        visited = set()

                        while pq:
                            time_so_far, _, node = heapq.heappop(pq)
                            if node in visited: continue
                            visited.add(node)

                            for neighbor, distance in graph[node]:
                                traffic = get_traffic_level(node, neighbor, time_period)
                                time = distance / adjusted_speed(traffic)
                                new_time = time_so_far + time
                    ```
                    """)

            with col2:
                st.subheader("🚧 Unoptimized (Default) Path")
                st.markdown("""
                - **Goal**: Provide a baseline reference without smart optimization
                - **Algorithm**: Basic path traversal (e.g., BFS or uniform cost)
                - **Best for**:
                  - Comparing benefits of optimization
                  - Offline planning where traffic data is unavailable
                  - Routes with fixed patterns (e.g., delivery loops)
                - **Key Features**:
                  - Ignores real-time congestion or road conditions
                  - Follows the shortest geographical route regardless of traffic
                  - May lead through high-traffic zones
                """)

                # Example calculation
                with st.expander("📊 Technical Example"):
                    st.markdown("""
                    ```python
                    def basic_unoptimized_route(graph, start, end):
                        pq = [(0, start)]  # (distance, node)
                        visited = set()

                        while pq:
                            dist, node = heapq.heappop(pq)
                            if node in visited: continue
                            visited.add(node)

                            for neighbor, distance in graph[node]:
                                heapq.heappush(pq, (dist + distance, neighbor))
                    ```
                    """)

            st.markdown("---")

            # Algorithm selection logic
            st.subheader("🧠 Routing Decision Logic")
            st.markdown("""
            Depending on the selected mode:

            - **Time-Optimized**:
              - A* algorithm for fast, intelligent routing (especially critical facilities)
              - Dijkstra's for general time-based pathfinding
            - **Unoptimized**:
              - Uses a simple baseline algorithm to show what happens *without* optimization

            ### ⚖️ Time vs. Simplicity Trade-off
            """)
            
            # Trade-off visualization
            fig, ax = plt.subplots()
            x = [0, 1]
            optimized = [1, 0]
            unoptimized = [0, 1]

            ax.plot(x, optimized, label='Time-Optimized', marker='o', color='green')
            ax.plot(x, unoptimized, label='Unoptimized', marker='s', color='gray')

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Optimized', 'Unoptimized'])
            ax.set_ylabel('Routing Focus')
            ax.set_title('Strategy Comparison: Optimization vs Simplicity')
            ax.legend()

            col1, col2, col3 = st.columns([1, 2, 1])  # center column is wider
            with col2:
                st.pyplot(fig)


            st.markdown("""
            ---

            ## 🕒 Time of Day Considerations

            Traffic behavior depends on the selected time period:

            """)

            # Time periods with descriptions
            time_periods = {
                "morning": "🌅 (7–10 AM): Business district rush — heavy inbound flow",
                "afternoon": "☀️ (12–4 PM): School dismissals and errands — moderate traffic",
                "evening": "🌇 (5–8 PM): Outbound rush hour — major congestion",
                "night": "🌃 (9 PM–6 AM): Light traffic — but possible roadwork or closures"
            }

            for period, desc in time_periods.items():
                st.markdown(f"- **{period.capitalize()}** {desc}")

            st.markdown("""
            ---

            ## 🚑 Emergency Routing

            If your route involves a **critical medical facility**:
            - ✅ Automatically switches to **Time-Optimized mode**
            - 🚓 Assumes right-of-way in traffic modeling
            - 🧠 Uses A* for smarter pathfinding
            """)


        # Subtab 3 - Analysis & Comparisons (formerly tab3)
        with path_finder_tabs[2]:
            st.title("Analysis & Comparisons")
            if st.button("Compare Routing Strategies", key="compare_button"):
                if start_id == goal_id:
                    st.warning("Start and destination locations are the same. No comparison needed.")
                try:
                    with st.spinner('Calculating different paths...'):
                        # Calculate time-optimized path
                        if is_priority_route:
                            path, travel_time = a_star_time_dependent(graph, traffic_flows, start_id, goal_id, time_period)
                        else:
                            path, travel_time = dijkstra_time_dependent(graph, traffic_flows, start_id, goal_id, time_period)

                        # Calculate congestion-optimized path
                        if is_priority_route:
                            cong_path, cong_travel_time = a_star_congestion_dependent(graph, traffic_flows, start_id, goal_id, time_period)
                        else:
                            cong_path, cong_travel_time = dijkstra_congestion_dependent(graph, traffic_flows, start_id, goal_id, time_period)


                        time_segments, time_distance, time_actual_time,time_congestion_score = get_path_details(
                        path, graph, traffic_flows_dict, time_period, id_to_name, is_priority_route
                        )
                            # Get details for congestion path - Ensure 4 values are unpacked
                        cong_segments, cong_distance, cong_actual_time,cong_congestion_score = get_path_details(
                        cong_path, graph, traffic_flows_dict, time_period, id_to_name, is_priority_route
                        )

                        # Load traffic flows from CSV for speed calculations
                        # This assumes traffic_flows_df is already loaded elsewhere
                        # If not, we would add code to load it here:
                        # traffic_flows_df = pd.read_csv("traffic_flows.csv")

                        # Convert traffic flow dataframe to dictionary for lookup
                        traffic_flows_lookup = {}
                        for _, row in Traffic_flows.iterrows():
                            road_id = row['RoadID']
                            from_node, to_node = road_id.split('-')
                            flows = [row['Morning'], row['Afternoon'], row['Evening'], row['Night']]
                            traffic_flows_lookup[(from_node, to_node)] = flows
                            # Also add reverse direction if the graph is undirected
                            traffic_flows_lookup[(to_node, from_node)] = flows

                        # Display the comparison
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.subheader("**⏱️ Time-Optimized Path**")
                            if path and time_actual_time != float('inf'):
                                st.metric("Travel Time", f"{time_actual_time:.1f} min")
                                st.metric("Distance", f"{time_distance:.2f} km")
                                st.metric("Congestion Score", f"{time_congestion_score:.0f}")
                            with st.expander("Show Route & Map (Time-Optimized)"):
                                route_names = [get_location_name(node, id_to_name) for node in path]
                                st.write(" → ".join(route_names))
                                m_time = plot_folium_path(path, Nodes, id_to_name)
                                if m_time:
                                    folium_static(m_time, height=300) # Smaller map for comparison view
                                else:
                                    st.warning("Map unavailable.")

                        with col2:
                            st.subheader("🚗 Un-Optimized Path")
                            if cong_path and cong_actual_time != float('inf'):
                                st.metric("Travel Time", f"{cong_actual_time:.1f} min")
                                st.metric("Distance", f"{cong_distance:.2f} km")
                                st.metric("Congestion Score", f"{cong_congestion_score:.0f}")
                            with st.expander("Show Route & Map (Un-Optimized Path)"):
                                opt_names = [get_location_name(node, id_to_name) for node in cong_path]
                                st.write(" → ".join(opt_names))
                                opt_map = plot_folium_path(cong_path, Nodes, id_to_name)
                                if opt_map:
                                    folium_static(opt_map, height=300) # Smaller map
                                else:
                                    st.warning("Map unavailable.")

                        with col3:
                            st.subheader("🚑 Emergency Route")
                            # Calculate emergency path using A* with emergency conditions
                            emergency_path, emergency_time = a_star_time_dependent(graph, traffic_flows, start_id, goal_id, time_period)
                            emergency_segments, emergency_distance, emergency_actual_time, emergency_congestion_score = get_path_details(
                                emergency_path, graph, traffic_flows_dict, time_period, id_to_name, True  # Always True for emergency routes
                            )
                            
                            if emergency_path and emergency_actual_time != float('inf'):
                                st.metric("Travel Time", f"{emergency_actual_time:.1f} min")
                                st.metric("Distance", f"{emergency_distance:.2f} km")
                                st.metric("Congestion Score", f"{emergency_congestion_score:.0f}")
                            with st.expander("Show Route & Map (Emergency Path)"):
                                emergency_names = [get_location_name(node, id_to_name) for node in emergency_path]
                                st.write(" → ".join(emergency_names))
                                emergency_map = plot_folium_path(emergency_path, Nodes, id_to_name)
                                if emergency_map:
                                    folium_static(emergency_map, height=300)
                                else:
                                    st.warning("Map unavailable.")

                        # Traffic Pattern Analysis
                        st.markdown("---")
                        st.subheader("📊 Traffic Pattern Analysis")
                        tcol1, tcol2, tcol3 = st.columns(3)

                        with tcol1:
                            if path and cong_path:
                                if path and len(path) > 1:
                                    # Create data for visualization
                                    time_periods = ["Morning", "Afternoon", "Evening", "Night"]
                                    traffic_by_period = []

                                    for i in range(len(path) - 1):
                                        from_node = str(path[i])
                                        to_node = str(path[i+1])
                                        road_id = f"{from_node}-{to_node}"
                                        alt_road_id = f"{to_node}-{from_node}"

                                        # Check CSV data for this road segment
                                        if road_id in Traffic_flows['RoadID'].values:
                                            row = Traffic_flows[Traffic_flows['RoadID'] == road_id].iloc[0]
                                            segment_name = f"{get_location_name(from_node, id_to_name)} to {get_location_name(to_node, id_to_name)}"
                                            for period in time_periods:
                                                traffic_by_period.append({
                                                    "Segment": segment_name,
                                                    "Time Period": period,
                                                    "Traffic Flow": row[period]
                                                })
                                        elif alt_road_id in Traffic_flows['RoadID'].values:
                                            # Try reverse direction if available
                                            row = Traffic_flows[Traffic_flows['RoadID'] == alt_road_id].iloc[0]
                                            segment_name = f"{get_location_name(from_node, id_to_name)} to {get_location_name(to_node, id_to_name)}"
                                            for period in time_periods:
                                                traffic_by_period.append({
                                                    "Segment": segment_name,
                                                    "Time Period": period,
                                                    "Traffic Flow": row[period]
                                                })

                                    if traffic_by_period:
                                        traffic_df = pd.DataFrame(traffic_by_period)
                                        st.bar_chart(traffic_df.pivot(index="Segment", columns="Time Period", values="Traffic Flow"))
                                    else:
                                        st.warning("No traffic data available for the selected path.")

                        with tcol2:
                            if path and cong_path:
                                if cong_path and len(cong_path) > 1:
                                    # Create data for visualization
                                    time_periods = ["Morning", "Afternoon", "Evening", "Night"]
                                    traffic_by_period = []

                                    for i in range(len(cong_path) - 1):
                                        from_node = str(cong_path[i])
                                        to_node = str(cong_path[i+1])
                                        road_id = f"{from_node}-{to_node}"
                                        alt_road_id = f"{to_node}-{from_node}"

                                        # Check CSV data for this road segment
                                        if road_id in Traffic_flows['RoadID'].values:
                                            row = Traffic_flows[Traffic_flows['RoadID'] == road_id].iloc[0]
                                            segment_name = f"{get_location_name(from_node, id_to_name)} to {get_location_name(to_node, id_to_name)}"
                                            for period in time_periods:
                                                traffic_by_period.append({
                                                    "Segment": segment_name,
                                                    "Time Period": period,
                                                    "Traffic Flow": row[period]
                                                })
                                        elif alt_road_id in Traffic_flows['RoadID'].values:
                                            # Try reverse direction if available
                                            row = Traffic_flows[Traffic_flows['RoadID'] == alt_road_id].iloc[0]
                                            segment_name = f"{get_location_name(from_node, id_to_name)} to {get_location_name(to_node, id_to_name)}"
                                            for period in time_periods:
                                                traffic_by_period.append({
                                                    "Segment": segment_name,
                                                    "Time Period": period,
                                                    "Traffic Flow": row[period]
                                                })

                                    if traffic_by_period:
                                        traffic_df = pd.DataFrame(traffic_by_period)
                                        st.bar_chart(traffic_df.pivot(index="Segment", columns="Time Period", values="Traffic Flow"))
                                    else:
                                        st.warning("No traffic data available for the selected path.")

                        with tcol3:
                            if emergency_path and len(emergency_path) > 1:
                                # Create data for visualization
                                time_periods = ["Morning", "Afternoon", "Evening", "Night"]
                                traffic_by_period = []

                                for i in range(len(emergency_path) - 1):
                                    from_node = str(emergency_path[i])
                                    to_node = str(emergency_path[i+1])
                                    road_id = f"{from_node}-{to_node}"
                                    alt_road_id = f"{to_node}-{from_node}"

                                    # Check CSV data for this road segment
                                    if road_id in Traffic_flows['RoadID'].values:
                                        row = Traffic_flows[Traffic_flows['RoadID'] == road_id].iloc[0]
                                        segment_name = f"{get_location_name(from_node, id_to_name)} to {get_location_name(to_node, id_to_name)}"
                                        for period in time_periods:
                                            traffic_by_period.append({
                                                "Segment": segment_name,
                                                "Time Period": period,
                                                "Traffic Flow": row[period]
                                            })
                                    elif alt_road_id in Traffic_flows['RoadID'].values:
                                        # Try reverse direction if available
                                        row = Traffic_flows[Traffic_flows['RoadID'] == alt_road_id].iloc[0]
                                        segment_name = f"{get_location_name(from_node, id_to_name)} to {get_location_name(to_node, id_to_name)}"
                                        for period in time_periods:
                                            traffic_by_period.append({
                                                "Segment": segment_name,
                                                "Time Period": period,
                                                "Traffic Flow": row[period]
                                            })

                                if traffic_by_period:
                                    traffic_df = pd.DataFrame(traffic_by_period)
                                    st.bar_chart(traffic_df.pivot(index="Segment", columns="Time Period", values="Traffic Flow"))
                                else:
                                    st.warning("No traffic data available for the emergency path.")

                        # Average Speed Analysis
                        st.markdown("---")
                        st.subheader("🏎 Average Speed Analysis")
                        scol1, scol2, scol3 = st.columns(3)

                        with scol1:
                            if path and cong_path:
                                if time_segments:
                                    # Calculate speed for each segment
                                    speeds = []

                                    for segment in time_segments:
                                        try:
                                            distance = float(segment["Distance (km)"])
                                            # Extract numeric part from time string
                                            time_min = float(segment["Time (min)"].split()[0])

                                            # Get node IDs for the segment
                                            from_node = segment.get("From ID", "")
                                            to_node = segment.get("To ID", "")
                                            if not from_node or not to_node:
                                                # Try to extract IDs from names if IDs not available
                                                # This is a fallback and depends on how segment data is structured
                                                from_name = segment.get("From", "")
                                                to_name = segment.get("To", "")
                                                for id, name in id_to_name.items():
                                                    if name == from_name:
                                                        from_node = id
                                                    if name == to_name:
                                                        to_node = id

                                            # Format road ID for CSV lookup
                                            road_id = f"{from_node}-{to_node}"
                                            alt_road_id = f"{to_node}-{from_node}"  # Try reverse direction too

                                            # Get traffic flow for current time period
                                            # Map time_period to column name in CSV
                                            period_map = {
                                                "morning": "Morning",
                                                "afternoon": "Afternoon",
                                                "evening": "Evening",
                                                "night": "Night"
                                            }
                                            period_col = period_map.get(time_period.lower(), "Morning")

                                            # Look up traffic flow in CSV data
                                            traffic_flow = 1000  # Default if not found
                                            if road_id in Traffic_flows['RoadID'].values:
                                                traffic_flow = Traffic_flows[Traffic_flows['RoadID'] == road_id][period_col].iloc[0]
                                            elif alt_road_id in Traffic_flows['RoadID'].values:
                                                traffic_flow = Traffic_flows[Traffic_flows['RoadID'] == alt_road_id][period_col].iloc[0]

                                            # Calculate base speed (km/h)
                                            base_speed = distance / (time_min / 60)

                                            # Apply congestion factor based on traffic flow
                                            # Higher traffic flow = lower speed
                                            # 4000 veh/h is considered heavy congestion
                                            congestion_factor = max(0.4, min(1.0, 1 - (traffic_flow / 8000)))
                                            adjusted_speed = base_speed * congestion_factor

                                            speeds.append({
                                                "Segment": f"{segment['From']} → {segment['To']}",
                                                "Speed (km/h)": adjusted_speed,
                                                "Traffic Flow (veh/h)": traffic_flow,
                                                "Distance (km)": distance,
                                                "Time (min)": time_min
                                            })
                                        except (ValueError, TypeError, ZeroDivisionError) as e:
                                            # Skip segments with invalid data
                                            continue
                                        
                                    if speeds:
                                        speeds_df = pd.DataFrame(speeds)
                                        avg_speed = speeds_df["Speed (km/h)"].mean()

                                        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
                                        st.bar_chart(speeds_df.set_index("Segment")["Speed (km/h)"])

                                        # Show speeds in a table for detailed view
                                        with st.expander("Detailed Speed Data"):
                                            st.dataframe(speeds_df)
                                    else:
                                        st.warning("Could not calculate speeds for the selected path.")
                        with scol2:
                            if path and cong_path:
                                if cong_segments:
                                    # Calculate speed for each segment
                                    speeds = []

                                    for segment in cong_segments:
                                        try:
                                            distance = float(segment["Distance (km)"])
                                            # Extract numeric part from time string
                                            time_min = float(segment["Time (min)"].split()[0])

                                            # Get node IDs for the segment
                                            from_node = segment.get("From ID", "")
                                            to_node = segment.get("To ID", "")
                                            if not from_node or not to_node:
                                                # Try to extract IDs from names if IDs not available
                                                from_name = segment.get("From", "")
                                                to_name = segment.get("To", "")
                                                for id, name in id_to_name.items():
                                                    if name == from_name:
                                                        from_node = id
                                                    if name == to_name:
                                                        to_node = id

                                            # Format road ID for CSV lookup
                                            road_id = f"{from_node}-{to_node}"
                                            alt_road_id = f"{to_node}-{from_node}"  # Try reverse direction too

                                            # Get traffic flow for current time period
                                            # Map time_period to column name in CSV
                                            period_map = {
                                                "morning": "Morning",
                                                "afternoon": "Afternoon",
                                                "evening": "Evening",
                                                "night": "Night"
                                            }
                                            period_col = period_map.get(time_period.lower(), "Morning")

                                            # Look up traffic flow in CSV data
                                            traffic_flow = 1000  # Default if not found
                                            if road_id in Traffic_flows['RoadID'].values:
                                                traffic_flow = Traffic_flows[Traffic_flows['RoadID'] == road_id][period_col].iloc[0]
                                            elif alt_road_id in Traffic_flows['RoadID'].values:
                                                traffic_flow = Traffic_flows[Traffic_flows['RoadID'] == alt_road_id][period_col].iloc[0]

                                            # Calculate base speed (km/h)
                                            base_speed = distance / (time_min / 60)

                                            # Apply congestion factor based on traffic flow
                                            # Higher traffic flow = lower speed
                                            # 4000 veh/h is considered heavy congestion
                                            congestion_factor = max(0.4, min(1.0, 1 - (traffic_flow / 8000)))
                                            adjusted_speed = base_speed * congestion_factor

                                            speeds.append({
                                                "Segment": f"{segment['From']} → {segment['To']}",
                                                "Speed (km/h)": adjusted_speed,
                                                "Traffic Flow (veh/h)": traffic_flow,
                                                "Distance (km)": distance,
                                                "Time (min)": time_min
                                            })
                                        except (ValueError, TypeError, ZeroDivisionError) as e:
                                            # Skip segments with invalid data
                                            continue
                                        
                                    if speeds:
                                        speeds_df = pd.DataFrame(speeds)
                                        avg_speed = speeds_df["Speed (km/h)"].mean()

                                        st.metric("Average Speed", f"{avg_speed:.1f} km/h")

                                        st.bar_chart(speeds_df.set_index("Segment")["Speed (km/h)"],color='#F1732E')

                                        # Show speeds in a table for detailed view
                                        with st.expander("Detailed Speed Data"):
                                            st.dataframe(speeds_df)
                                    else:
                                        st.warning("Could not calculate speeds for the selected path.")                        
                        with scol3:
                            if emergency_path and emergency_segments:
                                # Calculate speed for each segment
                                speeds = []

                                for segment in emergency_segments:
                                    try:
                                        distance = float(segment["Distance (km)"])
                                        # Extract numeric part from time string
                                        time_min = float(segment["Time (min)"].split()[0])

                                        # Get node IDs for the segment
                                        from_node = segment.get("From ID", "")
                                        to_node = segment.get("To ID", "")
                                        if not from_node or not to_node:
                                            # Try to extract IDs from names if IDs not available
                                            from_name = segment.get("From", "")
                                            to_name = segment.get("To", "")
                                            for id, name in id_to_name.items():
                                                if name == from_name:
                                                    from_node = id
                                                if name == to_name:
                                                    to_node = id

                                            # Format road ID for CSV lookup
                                            road_id = f"{from_node}-{to_node}"
                                            alt_road_id = f"{to_node}-{from_node}"  # Try reverse direction too

                                            # Get traffic flow for current time period
                                            # Map time_period to column name in CSV
                                            period_map = {
                                                "morning": "Morning",
                                                "afternoon": "Afternoon",
                                                "evening": "Evening",
                                                "night": "Night"
                                            }
                                            period_col = period_map.get(time_period.lower(), "Morning")

                                            # Look up traffic flow in CSV data
                                            traffic_flow = 1000  # Default if not found
                                            if road_id in Traffic_flows['RoadID'].values:
                                                traffic_flow = Traffic_flows[Traffic_flows['RoadID'] == road_id][period_col].iloc[0]
                                            elif alt_road_id in Traffic_flows['RoadID'].values:
                                                traffic_flow = Traffic_flows[Traffic_flows['RoadID'] == alt_road_id][period_col].iloc[0]

                                            # Calculate base speed (km/h)
                                            base_speed = distance / (time_min / 60)

                                            # Apply congestion factor based on traffic flow
                                            # Higher traffic flow = lower speed
                                            # 4000 veh/h is considered heavy congestion
                                            # Emergency vehicles get better congestion factor
                                            congestion_factor = max(0.6, min(1.0, 1 - (traffic_flow / 12000)))
                                            adjusted_speed = base_speed * congestion_factor

                                            speeds.append({
                                                "Segment": f"{segment['From']} → {segment['To']}",
                                                "Speed (km/h)": adjusted_speed,
                                                "Traffic Flow (veh/h)": traffic_flow,
                                                "Distance (km)": distance,
                                                "Time (min)": time_min
                                            })
                                    except (ValueError, TypeError, ZeroDivisionError) as e:
                                        # Skip segments with invalid data
                                        continue

                                if speeds:
                                    speeds_df = pd.DataFrame(speeds)
                                    avg_speed = speeds_df["Speed (km/h)"].mean()

                                    st.metric("Average Speed", f"{avg_speed:.1f} km/h")

                                    st.bar_chart(speeds_df.set_index("Segment")["Speed (km/h)"],color='#FF4B4B')

                                    # Show speeds in a table for detailed view
                                    with st.expander("Detailed Speed Data"):
                                        st.dataframe(speeds_df)
                                else:
                                    st.warning("Could not calculate speeds for the emergency path.")
                except Exception as e:
                    st.error(f"Error calculating comparison paths: {str(e)}")
                    st.exception(e)

    # Third tab - Public Transportation
    with tab3:
        pt_tabs = st.tabs(["🚞 Transportation Suggestions", "⏰ Schedule Optimization", "📊 Analysis"])
        # Load public transportation data
        @st.cache_data
        def load_pt_data():
            metro_lines = pd.read_csv('metro_lines.csv')
            bus_routes = pd.read_csv('bus_routes.csv')
            
            # Convert station IDs to strings and preserve facility IDs
            def convert_stations(stations_str):
                if isinstance(stations_str, str):
                    # Remove brackets and split by comma
                    stations = stations_str.strip('[]').split(',')
                    # Clean each station ID and convert to string
                    return [str(station.strip().strip("'").strip('"')) for station in stations]
                return stations_str
            
            # Apply conversion to both metro and bus data
            metro_lines['Stations'] = metro_lines['Stations'].apply(convert_stations)
            bus_routes['Stops'] = bus_routes['Stops'].apply(convert_stations)
            
            return metro_lines, bus_routes
        
        metro_lines, bus_routes = load_pt_data()

        def calculate_travel_time(mode, num_stops, time_period, start_id, end_id, traffic_flows, graph):
            """Calculate travel time based on mode, distance, and traffic conditions"""
            if mode == 'Metro':
                # Metro uses fixed high speed (200 km/h) and is not affected by traffic
                metro_speed = 120  # km/h
                
                # Use dijkstra to find the path and distance
                path, _ = dijkstra_time_dependent(graph, traffic_flows, start_id, end_id, time_period)
                
                if not path:
                    return float('inf')
                
                # Calculate total distance from the path
                total_distance = 0
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    for neighbor, distance in graph.get(current, []):
                        if neighbor == next_node:
                            total_distance += distance
                            break
                
                # Calculate time based on distance and fixed speed
                travel_time = (total_distance / metro_speed) * 60  # Convert to minutes
                
            else:  # Bus
                # Bus speed is affected by traffic
                base_speed = 40  # km/h (base speed without traffic)
                time_idx = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}[time_period]
                
                # Use dijkstra to find the path
                path, _ = dijkstra_time_dependent(graph, traffic_flows, start_id, end_id, time_period)
                
                if not path:
                    return float('inf')
                
                # Calculate total distance and traffic impact
                total_distance = 0
                total_traffic_impact = 0
                
                for i in range(len(path) - 1):
                    current = path[i]
                    next_node = path[i + 1]
                    for neighbor, distance in graph.get(current, []):
                        if neighbor == next_node:
                            total_distance += distance
                            # Get traffic flow for this segment
                            edge = (current, next_node)
                            if edge in traffic_flows:
                                traffic_flow = traffic_flows[edge][time_idx]
                                # Calculate traffic impact (higher flow = lower speed)
                                traffic_impact = min(traffic_flow / 4000, 0.8)  # Cap at 80% reduction
                                total_traffic_impact += traffic_impact * distance
                            break
                
                # Calculate average traffic impact
                avg_traffic_impact = total_traffic_impact / total_distance if total_distance > 0 else 0
                
                # Adjust speed based on traffic
                adjusted_speed = base_speed * (1 - avg_traffic_impact)
                
                # Calculate time based on distance and adjusted speed
                travel_time = (total_distance / adjusted_speed) * 60  # Convert to minutes
                
                # Add time for stops (30 seconds per stop)
                travel_time += (num_stops * 0.5)
            
            return round(travel_time, 1)

        def suggest_transportation(start_id, end_id, time_period, metro_lines, bus_routes):
            """Suggest the best public transportation option based on user inputs"""
            # Convert IDs to strings for consistency
            start_id = str(start_id)
            end_id = str(end_id)
            
            # Initialize results
            suggestions = []
            
            # Get location names
            start_name = id_to_name.get(start_id, start_id)
            end_name = id_to_name.get(end_id, end_id)
            
            # Check for direct metro connection
            for _, metro in metro_lines.iterrows():
                stations = metro['Stations']  # Already a list
                if start_id in stations and end_id in stations:
                    # Check if stations are in correct order
                    start_idx = stations.index(start_id)
                    end_idx = stations.index(end_id)
                    if start_idx < end_idx:
                        num_stations = len(stations[start_idx:end_idx+1])
                        travel_time = calculate_travel_time('Metro', num_stations, time_period, start_id, end_id, traffic_flows_dict, graph)
                        
                        # Calculate actual distance using the graph
                        total_distance = 0
                        for i in range(start_idx, end_idx):
                            current = stations[i]
                            next_station = stations[i+1]
                            for neighbor, distance in graph.get(current, []):
                                if neighbor == next_station:
                                    total_distance += distance
                                    break
                        
                        suggestions.append({
                            'Type': 'Metro',
                            'Line': metro['Name'],
                            'Stations': num_stations,
                            'Direct': True,
                            'Time': travel_time,
                            'Distance': total_distance,
                            'Details': f"Take {metro['Name']} from {start_name} to {end_name}"
                        })
            
            # Check for direct bus connection (bidirectional)
            for _, bus in bus_routes.iterrows():
                stops = bus['Stops']  # Already a list
                if start_id in stops and end_id in stops:
                    start_idx = stops.index(start_id)
                    end_idx = stops.index(end_id)
                    
                    # Check both directions
                    if start_idx < end_idx:
                        # Forward direction
                        num_stops = len(stops[start_idx:end_idx+1])
                        travel_time = calculate_travel_time('Bus', num_stops, time_period, start_id, end_id, traffic_flows_dict, graph)
                        
                        # Calculate actual distance using the graph
                        total_distance = 0
                        for i in range(start_idx, end_idx):
                            current = stops[i]
                            next_stop = stops[i+1]
                            for neighbor, distance in graph.get(current, []):
                                if neighbor == next_stop:
                                    total_distance += distance
                                    break
                        
                        suggestions.append({
                            'Type': 'Bus',
                            'Route': f"Route {bus['RouteID']}",
                            'Stops': num_stops,
                            'Direct': True,
                            'Time': travel_time,
                            'Distance': total_distance,
                            'Details': f"Take Bus Route {bus['RouteID']} from {start_name} to {end_name}"
                        })
                    else:
                        # Reverse direction
                        num_stops = len(stops[end_idx:start_idx+1])
                        travel_time = calculate_travel_time('Bus', num_stops, time_period, end_id, start_id, traffic_flows_dict, graph)
                        
                        # Calculate actual distance using the graph
                        total_distance = 0
                        for i in range(end_idx, start_idx):
                            current = stops[i]
                            next_stop = stops[i+1]
                            for neighbor, distance in graph.get(current, []):
                                if neighbor == next_stop:
                                    total_distance += distance
                                    break
                        
                        suggestions.append({
                            'Type': 'Bus',
                            'Route': f"Route {bus['RouteID']}",
                            'Stops': num_stops,
                            'Direct': True,
                            'Time': travel_time,
                            'Distance': total_distance,
                            'Details': f"Take Bus Route {bus['RouteID']} from {start_name} to {end_name} (reverse direction)"
                        })
            
            # Sort suggestions by travel time
            suggestions.sort(key=lambda x: x['Time'])
            return suggestions

        with pt_tabs[0]:
            st.title("Transportation Suggestions")
            if st.button("Get Transportation Suggestions", key="get_transport_suggestions"):
                # Get user inputs from the sidebar
                start_location = st.session_state.start_location
                goal_location = st.session_state.goal_location
                time_period = st.session_state.time_period_radio
                
                if start_location and goal_location:
                    start_id = name_to_id.get(start_location.lower())
                    goal_id = name_to_id.get(goal_location.lower())
                    
                    if start_id and goal_id:
                        suggestions = suggest_transportation(start_id, goal_id, time_period, metro_lines, bus_routes)
                        
                        if suggestions:
                            st.success("Found transportation options!")
                            
                            # Display suggestions in a clean format
                            for i, suggestion in enumerate(suggestions, 1):
                                st.markdown(f"### Option {i}: {suggestion['Type']} Route")
                                
                                if suggestion['Direct']:
                                    if suggestion['Type'] == 'Metro':
                                        st.markdown(f"🚇 **Metro Line:** {suggestion['Line']}")
                                        st.markdown(f"📊 **Number of stations:** {suggestion['Stations']}")
                                    else:
                                        st.markdown(f"🚌 **Bus Route:** {suggestion['Route']}")
                                        st.markdown(f"🚏 **Number of stops:** {suggestion['Stops']}")
                                else:
                                    st.markdown(f"🔄 **Transfer Point:** {suggestion['Transfer Point']}")
                                
                                st.markdown(f"⏱️ **Estimated Travel Time:** {suggestion['Time']} minutes")
                                st.markdown(f"📏 **Total Distance:** {suggestion.get('Distance', 0):.2f} km")
                                st.markdown(f"**Route Details:** {suggestion['Details']}")
                                
                                if i < len(suggestions):
                                    st.markdown("---")
                        else:
                            st.warning("No public transportation options available for this route.")
                    else:
                        st.error("Could not find transportation information for the selected locations.")
                else:
                    st.info("Please select start and destination locations in the sidebar.")
        
        with pt_tabs[1]:
            
            
            st.title("Schedule Optimization")
            # Display route details in a simple and good looking way
            bcol1,bcol2=st.columns(2)
            with bcol1:
                st.header("🚍 Bus Routes")
                # Create a base map for bus routes
                bus_map = folium.Map(location=[30.0444, 31.2357], zoom_start=11, tiles='CartoDB positron')
                # Define colors for bus routes
                bus_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                for idx, bus in bus_routes.iterrows():
                    with st.expander(f"Bus Route {bus['RouteID']}"):
                        stops = bus['Stops']
                        stop_names = [id_to_name.get(stop, f"Unknown (ID: {stop})") for stop in stops]
                        st.markdown("**Stops:**")
                        for i, name in enumerate(stop_names, 1):
                            st.markdown(f"{i}. {name}")
                        # Add route to map
                        if len(stops) > 1:
                            path_coords = []
                            for stop in stops:
                                stop_id = str(stop)
                                if stop_id in coordinates:
                                    lon, lat = coordinates[stop_id]
                                    path_coords.append([lat, lon])
                            if len(path_coords) > 1:
                                # Draw bus route line with unique color
                                route_color = bus_colors[idx % len(bus_colors)]
                                folium.PolyLine(
                                    path_coords,
                                    color=route_color,
                                    weight=3,
                                    opacity=0.8,
                                    tooltip=f"Bus Route {bus['RouteID']}"
                                ).add_to(bus_map)
                                # Add markers for stops
                                for i, coord in enumerate(path_coords):
                                    stop_id = str(stops[i])
                                    stop_name = id_to_name.get(stop_id, f"Unknown (ID: {stop_id})")
                                    tooltip_html = f"""
                                    <div style="font-family: Arial; width: 200px;">
                                        <h4 style="margin: 0;">{stop_name}</h4>
                                        <p style="margin: 0;"><b>Type:</b> Bus Stop</p>
                                        <p style="margin: 0;"><b>Route:</b> {bus['RouteID']}</p>
                                    </div>
                                    """
                                    folium.CircleMarker(
                                        coord,
                                        radius=5,
                                        color=route_color,
                                        fill=True,
                                        fill_opacity=0.7,
                                        tooltip=folium.Tooltip(tooltip_html)
                                    ).add_to(bus_map)
                                    folium.map.Marker(
                                        coord,
                                        icon=DivIcon(
                                            icon_size=(120, 36),
                                            icon_anchor=(60, -10),
                                            html=f'<div style="font-size: 8pt; color: black; font-weight: bold; text-align: center;">{stop_name}</div>'
                                        )
                                    ).add_to(bus_map)
                        st.markdown("---")
            with bcol2:
            # Display the bus routes map
                st.header("🌍 Bus Map")
                folium_static(bus_map, width=800, height=600)
            st.markdown("---")
            mcol1,mcol2=st.columns(2)
            with mcol1:
                st.header("🚆 Metro Lines")
                # Create a base map for metro lines
                metro_map = folium.Map(location=[30.0444, 31.2357], zoom_start=11, tiles='CartoDB positron')

                # Define colors for metro lines
                metro_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

                for idx, metro in metro_lines.iterrows():
                    with st.expander(f"Metro Line {metro['Name']}"):
                        stations = metro['Stations']
                        station_names = [id_to_name.get(station, f"Unknown (ID: {station})") for station in stations]
                        st.markdown("**Stations:**")
                        for i, name in enumerate(station_names, 1):
                            st.markdown(f"{i}. {name}")

                        # Add route to map
                        if len(stations) > 1:
                            path_coords = []
                            for station in stations:
                                station_id = str(station)
                                if station_id in coordinates:
                                    lon, lat = coordinates[station_id]
                                    path_coords.append([lat, lon])

                            if len(path_coords) > 1:
                                # Draw metro line with unique color
                                line_color = metro_colors[idx % len(metro_colors)]
                                folium.PolyLine(
                                    path_coords,
                                    color=line_color,
                                    weight=4,
                                    opacity=0.8,
                                    tooltip=f"Metro Line {metro['Name']}"
                                ).add_to(metro_map)

                                # Add markers for stations
                                for i, coord in enumerate(path_coords):
                                    station_id = str(stations[i])
                                    station_name = id_to_name.get(station_id, f"Unknown (ID: {station_id})")

                                    tooltip_html = f"""
                                    <div style="font-family: Arial; width: 200px;">
                                        <h4 style="margin: 0;">{station_name}</h4>
                                        <p style="margin: 0;"><b>Type:</b> Metro Station</p>
                                        <p style="margin: 0;"><b>Line:</b> {metro['Name']}</p>
                                    </div>
                                    """

                                    folium.CircleMarker(
                                        coord,
                                        radius=6,
                                        color=line_color,
                                        fill=True,
                                        fill_opacity=0.7,
                                        tooltip=folium.Tooltip(tooltip_html)
                                    ).add_to(metro_map)

                                    folium.map.Marker(
                                        coord,
                                        icon=DivIcon(
                                            icon_size=(120, 36),
                                            icon_anchor=(60, -10),
                                            html=f'<div style="font-size: 8pt; color: black; font-weight: bold; text-align: center;">{station_name}</div>'
                                        )
                                    ).add_to(metro_map)
            with mcol2:
                st.header("🌍 Metro Map")
                folium_static(metro_map, width=800, height=600)
        with pt_tabs[2]:
            st.title("Public Transportation Analysis")
            
            # Create columns for different analysis sections
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.subheader("🚌 Bus Network Analysis")
                
                # Calculate bus network statistics
                total_bus_routes = len(bus_routes)
                total_bus_stops = sum(len(route['Stops']) for _, route in bus_routes.iterrows())
                avg_stops_per_route = total_bus_stops / total_bus_routes if total_bus_routes > 0 else 0
                
                # Display bus network metrics
                st.metric("Total Bus Routes", total_bus_routes)
                st.metric("Total Bus Stops", total_bus_stops)
                st.metric("Average Stops per Route", f"{avg_stops_per_route:.1f}")
                
                # Bus route length analysis
                route_lengths = []
                for _, route in bus_routes.iterrows():
                    stops = route['Stops']
                    total_distance = 0
                    for i in range(len(stops) - 1):
                        current = str(stops[i])
                        next_stop = str(stops[i + 1])
                        for neighbor, distance in graph.get(current, []):
                            if neighbor == next_stop:
                                total_distance += distance
                                break
                    route_lengths.append(total_distance)
                
                if route_lengths:
                    avg_route_length = sum(route_lengths) / len(route_lengths)
                    st.metric("Average Route Length", f"{avg_route_length:.1f} km")
                    
                    # Create histogram of route lengths
                    fig = px.histogram(
                        x=route_lengths,
                        title="Distribution of Bus Route Lengths",
                        labels={"x": "Route Length (km)", "y": "Number of Routes"},
                        nbins=10
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Bus stop frequency analysis
                stop_frequency = {}
                for _, route in bus_routes.iterrows():
                    for stop in route['Stops']:
                        stop_id = str(stop)
                        stop_frequency[stop_id] = stop_frequency.get(stop_id, 0) + 1
                
                if stop_frequency:
                    most_common_stops = sorted(stop_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.subheader("Most Connected Bus Stops")
                    for stop_id, frequency in most_common_stops:
                        stop_name = id_to_name.get(stop_id, f"Unknown (ID: {stop_id})")
                        st.markdown(f"- {stop_name}: {frequency} routes")
            
            with analysis_col2:
                st.subheader("🚇 Metro Network Analysis")
                
                # Calculate metro network statistics
                total_metro_lines = len(metro_lines)
                total_metro_stations = sum(len(line['Stations']) for _, line in metro_lines.iterrows())
                avg_stations_per_line = total_metro_stations / total_metro_lines if total_metro_lines > 0 else 0
                
                # Display metro network metrics
                st.metric("Total Metro Lines", total_metro_lines)
                st.metric("Total Metro Stations", total_metro_stations)
                st.metric("Average Stations per Line", f"{avg_stations_per_line:.1f}")
                
                # Metro line length analysis
                line_lengths = []
                for _, line in metro_lines.iterrows():
                    stations = line['Stations']
                    total_distance = 0
                    for i in range(len(stations) - 1):
                        current = str(stations[i])
                        next_station = str(stations[i + 1])
                        for neighbor, distance in graph.get(current, []):
                            if neighbor == next_station:
                                total_distance += distance
                                break
                    line_lengths.append(total_distance)
                
                if line_lengths:
                    avg_line_length = sum(line_lengths) / len(line_lengths)
                    st.metric("Average Line Length", f"{avg_line_length:.1f} km")
                    
                    # Create histogram of line lengths
                    fig = px.histogram(
                        x=line_lengths,
                        title="Distribution of Metro Line Lengths",
                        labels={"x": "Line Length (km)", "y": "Number of Lines"},
                        nbins=5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Metro station frequency analysis
                station_frequency = {}
                for _, line in metro_lines.iterrows():
                    for station in line['Stations']:
                        station_id = str(station)
                        station_frequency[station_id] = station_frequency.get(station_id, 0) + 1
                
                if station_frequency:
                    most_common_stations = sorted(station_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
                    st.subheader("Most Connected Metro Stations")
                    for station_id, frequency in most_common_stations:
                        station_name = id_to_name.get(station_id, f"Unknown (ID: {station_id})")
                        st.markdown(f"- {station_name}: {frequency} lines")
            
            # Network Coverage Analysis
            st.markdown("---")
            st.subheader("🌍 Network Coverage Analysis")
            
            # Calculate coverage metrics
            all_bus_stops = set()
            all_metro_stations = set()
            
            for _, route in bus_routes.iterrows():
                all_bus_stops.update(str(stop) for stop in route['Stops'])
            
            for _, line in metro_lines.iterrows():
                all_metro_stations.update(str(station) for station in line['Stations'])
            
            total_locations = len(coordinates)
            bus_coverage = len(all_bus_stops) / total_locations * 100
            metro_coverage = len(all_metro_stations) / total_locations * 100
            combined_coverage = len(all_bus_stops.union(all_metro_stations)) / total_locations * 100
            
            # Display coverage metrics
            coverage_col1, coverage_col2, coverage_col3 = st.columns(3)
            with coverage_col1:
                st.metric("Bus Network Coverage", f"{bus_coverage:.1f}%")
            with coverage_col2:
                st.metric("Metro Network Coverage", f"{metro_coverage:.1f}%")
            with coverage_col3:
                st.metric("Combined Network Coverage", f"{combined_coverage:.1f}%")
            
            # Create coverage visualization
            coverage_data = {
                'Network Type': ['Bus', 'Metro', 'Combined'],
                'Coverage (%)': [bus_coverage, metro_coverage, combined_coverage]
            }
            coverage_df = pd.DataFrame(coverage_data)
            
            fig = px.bar(
                coverage_df,
                x='Network Type',
                y='Coverage (%)',
                title="Network Coverage Comparison",
                color='Network Type',
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations section
            st.markdown("---")
            st.subheader("💡 Recommendations")
            # Analyze gaps in coverage
            all_covered_locations = all_bus_stops.union(all_metro_stations)
            uncovered_locations = set(coordinates.keys()) - all_covered_locations
            
            if uncovered_locations:
                st.markdown("### Areas Needing Coverage")
                st.markdown("The following areas currently lack public transportation coverage:")
                for loc_id in uncovered_locations:
                    loc_name = id_to_name.get(loc_id, f"Unknown (ID: {loc_id})")
                    st.markdown(f"- {loc_name}")
            # Travel Time Comparison Analysis
            st.markdown("---")
            st.subheader("⏱️ Travel Time Comparison: Bus vs Metro")
            
            # Calculate average travel times for different time periods
            time_periods = ['morning', 'afternoon', 'evening', 'night']
            bus_times = []
            metro_times = []
            
            # Sample routes for comparison (using existing routes)
            sample_routes = []
            
            # Get bus routes
            for _, bus in bus_routes.iterrows():
                stops = bus['Stops']
                if len(stops) >= 2:  # Only consider routes with at least 2 stops
                    sample_routes.append(('bus', stops[0], stops[-1]))
            
            # Get metro routes
            for _, metro in metro_lines.iterrows():
                stations = metro['Stations']
                if len(stations) >= 2:  # Only consider lines with at least 2 stations
                    sample_routes.append(('metro', stations[0], stations[-1]))
            
            # Calculate travel times for each period
            for period in time_periods:
                period_bus_times = []
                period_metro_times = []
                
                for mode, start, end in sample_routes:
                    if mode == 'bus':
                        # Calculate bus travel time
                        path, _ = dijkstra_time_dependent(graph, traffic_flows_dict, str(start), str(end), period)
                        if path:
                            segments, _, time, _ = get_path_details(path, graph, traffic_flows_dict, period, id_to_name)
                            if time != float('inf'):
                                period_bus_times.append(time)
                    else:
                        # Calculate metro travel time
                        path, _ = dijkstra_time_dependent(graph, traffic_flows_dict, str(start), str(end), period)
                        if path:
                            segments, _, time, _ = get_path_details(path, graph, traffic_flows_dict, period, id_to_name)
                            if time != float('inf'):
                                period_metro_times.append(time)
                
                if period_bus_times:
                    bus_times.append(sum(period_bus_times) / len(period_bus_times))
                if period_metro_times:
                    metro_times.append(sum(period_metro_times) / len(period_metro_times))
            
            # Create comparison visualization
            if bus_times and metro_times:
                # Create DataFrame for plotting
                comparison_data = {
                    'Time Period': time_periods * 2,
                    'Mode': ['Bus'] * len(time_periods) + ['Metro'] * len(time_periods),
                    'Average Time (min)': bus_times + metro_times
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create bar chart
                fig = px.bar(
                    comparison_df,
                    x='Time Period',
                    y='Average Time (min)',
                    color='Mode',
                    barmode='group',
                    title='Average Travel Time Comparison: Bus vs Metro',
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics
                time_col1, time_col2, time_col3 = st.columns(3)
                
                with time_col1:
                    avg_bus_time = sum(bus_times) / len(bus_times)
                    st.metric("Average Bus Travel Time", f"{avg_bus_time:.1f} min")
                
                with time_col2:
                    avg_metro_time = sum(metro_times) / len(metro_times)
                    st.metric("Average Metro Travel Time", f"{avg_metro_time:.1f} min")
                
                with time_col3:
                    time_diff = ((avg_bus_time - avg_metro_time) / avg_bus_time) * 100
                    st.metric("Time Difference", f"{time_diff:.1f}% faster by metro")
                
                
if __name__ == '__main__':
    run_gui()   