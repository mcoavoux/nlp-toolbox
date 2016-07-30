import sys
import json
import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from bokeh.plotting import figure, output_file, show
from bokeh.charts import Histogram,Bar
from bokeh.io import gridplot
from bokeh.resources import CDN
from bokeh.embed import file_html

def plot_distance_distrib(G):
    """
    Plots the distribution of distances from the Graph
    """
    W = list([G[x][y]['weight'] for (x,y) in G.edges()])
    h = Histogram(W, title="Weight (proximity) distribution",xlabel='weights',ylabel="Counts(weights)",bins=50)
    return h
    
def plot_sem_distrib(G):
    """
    Plots the distribution of semantic classes from the Graph
    """
    W = pd.Series([G.node[x]['group'] for x in G.nodes()])
    C = W.value_counts(sort=False)
    p = Bar(C,xlabel="Semantic class", ylabel="Counts", title="Proportions of semantic classes")
    return p


def plot_node_degrees(G):
    """
    Plots the distribution of semantic classes from the Graph
    """
    W = pd.Series([G.degree(x) for x in G.nodes()])
    C = W.value_counts(sort=True,ascending=False)
    p = Bar(C, xlabel="Node Degree", ylabel="Counts", title="Distribution of node degrees")
    return p

def make_html_plots(G,outdir):

    px = plot_distance_distrib(G)
    py = plot_sem_distrib(G)
    pz = plot_node_degrees(G)
    p = gridplot([[px, py], [pz, None]])
    html = file_html(p, CDN, "Distance distribution")
    out = open(outdir+'/plots.html','w')
    out.write(html)
    out.close()

def read_json_graph(istream):
    """
    Reads a json graph output by the algorithm and returns it
    """
    data = json.loads(istream.read())
    G = json_graph.node_link_graph(data)
    return G

def postprocess_graph(G,istream):
    """
    Adds features to nodes from the istream original data
    """
    for idx,line in enumerate(istream):
        n = G.nodes()[idx]
        toks = line.split()
        G.node[idx]["example"] = toks[2]
        G.node[idx]["id"] = toks[1]
        G.node[idx]["sem"] = '['+','.join([ '%.2f'%(x,) for x in G.node[idx]["sem"]])+']'
        G.node[idx]["seed"] = True
        if toks[0] == "?":
            G.node[idx]["seed"] = False

            
def dump_graph(G,ostream):
    """
    dumps the postprocessed graph
    """
    print >> ostream, json.dumps(json_graph.node_link_data(G))
    
if __name__ == '__main__':
    infile = sys.argv[1]
    outdir = sys.argv[2]
    inGraph = open(outdir+"/wsd.json")
    inData = open(infile)
    outGraph = open(outdir+'/wsd-full.json',"w")
    G = read_json_graph(inGraph)    
    postprocess_graph(G,inData)
    dump_graph(G,outGraph)    
    inGraph.close()
    inData.close()
    outGraph.close()
    make_html_plots(G,outdir)
    

    
