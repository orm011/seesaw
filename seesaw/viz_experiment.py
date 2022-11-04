from torch.nn import functional as F
import torch
import torch.distributions as dist
import plotly.graph_objects as go
from .knn_graph import compute_exact_knn, get_weight_matrix, rbf_kernel
from sklearn.decomposition import PCA

class MixtureModelDistribution:
    def __init__(self,*, n_dim, n_classes, unit_length=True):
        
        unit_mu = F.normalize(torch.ones(n_dim), dim=-1)
        spread = unit_mu.min()/10
        meta_dist = dist.MultivariateNormal(loc=unit_mu, covariance_matrix=(spread**2)*torch.eye(n_dim))
        self.meta_dist = meta_dist
        self.mus = meta_dist.sample(sample_shape=(n_classes,))
        self.mus = F.normalize(self.mus, dim=-1)
        
        covars = torch.stack([torch.eye(n_dim)*((spread/2)**2) for i in range(n_classes)])
        self.class_conditional_dist = [dist.MultivariateNormal(loc=self.mus[i], covariance_matrix=covars[i]) 
                                        for i in range(n_classes)]
        
        class_probs = dist.Dirichlet(torch.ones(n_classes)).sample()
        class_probs = torch.sort(class_probs)[0] # - is the least popular
        self.class_dist = dist.Categorical(class_probs)

        
    def sample(self, *, n_samples):
        cats = self.class_dist.sample((n_samples,))
        points = torch.stack([self.class_conditional_dist[c].sample() for c in cats])
        points = F.normalize(points)
        return points.numpy(), cats.numpy()
        

class GraphPlot:
    def __init__(self, X, labels):
        self.pca = PCA(n_components=2)
        self.X = X
        self.Xplot = self.pca.fit_transform(self.X)

        self.labels = labels
        self.knng = compute_exact_knn(X, n_neighbors=3, metric='dot')
        self._init_plot_objects()
        
    def _init_plot_objects(self):
        ## want to show what happens to pseudo labels after finding a negative result
        # only used for making plot. use a different one for your own parameters
        Xplot = self.Xplot
        tmp_wmat = get_weight_matrix(self.knng, rbf_kernel(.2), self_edges=False)
        edge_x = []
        edge_y = []
        iis, jjs = tmp_wmat.nonzero()
        for i,j in zip(iis,jjs):
            if i >= j:
                continue

            x0, y0 = Xplot[i,0], Xplot[i,1]
            x1, y1 = Xplot[j,0], Xplot[j,1]    
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        self.edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for vec in self.Xplot:
            x, y = vec[0], vec[1]
            node_x.append(x)
            node_y.append(y)

        self.node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Score',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))
        
        self.node_trace.text = np.arange(self.X.shape[0])

    def _show_plot(self, fig_data=[]):  
        fig = go.Figure(data=[self.edge_trace, self.node_trace] + fig_data,
             layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
        return fig.show()
        
        
    def plot_vector(self, vector):
        if len(vector.shape) == 1:
            vector = vector.reshape(1,-1)
            
        x = self.pca.transform(vector)
        
        vec_node_trace = go.Scatter(
            x=x[:,0], y=x[:,1],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                # showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                # colorscale='',
                # reversescale=True,
                color=['red'],
                size=12,
                # colorbar=dict(
                #     thickness=15,
                #     title='Node Connections',
                #     xanchor='left',
                #     titleside='right'
                # ),
                line_width=3))
    
        new_labels = self.X @ vector.reshape(-1)
        self.node_trace.marker.color = new_labels
        return self._show_plot([vec_node_trace])
        
    def plot_labels(self, label_values=None):
        if label_values is None:
            label_values = self.labels
            
        self.node_trace.marker.color = label_values
        return self._show_plot()
