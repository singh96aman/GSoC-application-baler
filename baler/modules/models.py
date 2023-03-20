import torch
from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):
    def __init__(self, device, n_features, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device
        self.n_features = n_features
        self.z_dim = z_dim

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float64, device=device)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64, device=device)

        self.en3 = nn.Linear(100, 50, dtype=torch.float64, device=device)
        
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64, device=device)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64, device=device)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64, device=device)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float64, device=device)

        #latent Distribution
        self.fc_mu = nn.Linear(50, z_dim, dtype=torch.float64, device=device)
        self.fc_var = nn.Linear(50, z_dim, dtype=torch.float64, device=device)

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = self.en3(h2)
        mu = self.fc_mu(h3)
        log_var = self.fc_var(h3)
        return [mu, log_var]

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def reparameterize_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize_sample(mu, log_var)
        return  [self.decode(z), mu, log_var]

class CNN_VAE(nn.Module):
    def __init__(self, device, n_features, z_dim, *args, **kwargs):
        self.n_features = n_features
        super().__init__(*args, **kwargs)

        self.device = device
        self.n_features = n_features
        self.z_dim = z_dim

        # encoder
        self.en1 = nn.Conv1d(n_features, out_channels=64,
                              kernel_size= 3, stride= 2, padding  = 1, dtype=torch.float64, device=device)
        self.b1 = nn.BatchNorm1d(64, dtype=torch.float64, device=device)
        self.en2 = nn.Conv1d(64, out_channels=128,
                              kernel_size= 3, stride= 2, padding  = 1, dtype=torch.float64, device=device)
        self.b2 = nn.BatchNorm1d(128, dtype=torch.float64, device=device)
        self.en3 = nn.Conv1d(128, out_channels=256,
                              kernel_size= 3, stride= 2, padding  = 1, dtype=torch.float64, device=device)
        self.b3 = nn.BatchNorm1d(256, dtype=torch.float64, device=device)         
        
        # decoder
        self.de1 = nn.Linear(z_dim, 256, dtype=torch.float64, device=device)   
        self.de2 = nn.ConvTranspose1d(256, 128, stride=2, padding=1, output_padding=1, kernel_size= 3, dtype=torch.float64, device=device)
        self.de3 = nn.ConvTranspose1d(128, 64, stride=2, padding=1, output_padding=1, kernel_size= 3, dtype=torch.float64, device=device)
        self.de5 = nn.Conv1d(64, n_features, kernel_size= 3, stride= 2, padding  = 1, dtype=torch.float64, device=device)

        self.fc_mu = nn.Linear(256, z_dim, dtype=torch.float64, device=device)
        self.fc_var = nn.Linear(256, z_dim, dtype=torch.float64, device=device)

    def encode(self, x):
        x = x.resize(x.shape[0],self.n_features,1)
        h1 = F.leaky_relu(self.en1(x))
        h1 = self.b1(h1)
        h2 = F.leaky_relu(self.en2(h1))
        h2 = self.b2(h2)
        h3 = self.en3(h2)
        h3 = self.b3(h3)
        h3_flat=torch.flatten(h3,start_dim=1)
        mu = self.fc_mu(h3_flat)
        log_var = self.fc_var(h3_flat)
        return [mu, log_var]

    def decode(self, z):
        z=z.view(-1,1,4)
        h4 = F.leaky_relu(self.de1(z))
        h4 = h4.resize(z.shape[0], 256, 1)
        h5 = self.de2(h4)
        h6 = self.de3(h5)
        out = self.de5(h6)
        out = F.tanh(out)
        out = torch.mean(out, -1)
        return out

    def reparameterize_sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize_sample(mu, log_var)
        return  [self.decode(z), mu, log_var]

class george_SAE(nn.Module):
    def __init__(self, device, n_features, z_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = device

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float64, device=device)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64, device=device)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64, device=device)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float64, device=device)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64, device=device)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64, device=device)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64, device=device)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float64, device=device)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_BN(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_BN, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(z_dim, dtype=torch.float64),
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            # nn.BatchNorm1d(n_features,dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_Dropout_BN(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_Dropout_BN, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(z_dim,dtype=torch.float64)
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.BatchNorm1d(n_features, dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class george_SAE_Dropout(nn.Module):
    def __init__(self, n_features, z_dim):
        super(george_SAE_Dropout, self).__init__()

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(50, 100, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(100, 200, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.Linear(200, n_features, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        # z = x.view(batch_size,a,b,c) ? What is this
        return self.decode(x)

    def loss(self, model_children, true_data, reconstructed_data, reg_param):
        mse = nn.MSELoss()
        mse_loss = mse(reconstructed_data, true_data)
        l1_loss = 0
        values = true_data
        for i in range(len(model_children)):
            values = F.relu((model_children[i](values)))
            l1_loss += torch.mean(torch.abs(values))
        loss = mse_loss + reg_param * l1_loss
        return loss
