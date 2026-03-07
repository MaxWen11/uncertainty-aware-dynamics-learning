import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

# ==========================================
# Hyperparameter Configuration
# ==========================================
CONFIG = {
    'seed': 1011,
    'epochs': 2000,
    'batch_size': 1024,
    'lr_g': 5e-4,
    'lr_d': 5e-4,
    'gan_type': 'GPGAN',  # Options: 'LSGAN' or 'GPGAN'
    'lambda_gp': 10.0,    # Gradient penalty weight
    'lambda_anchor': 1.0  # Physical MSE anchor loss weight
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# Setup device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("MPS (Metal) GPU enabled.")
else:
    device = torch.device('cpu')
    print("No GPU detected, using CPU.")

# ==========================================
# 1. Gradient Penalty Function
# ==========================================
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones(real_samples.size(0), 1).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ==========================================
# 2. Neural Network Definition
# ==========================================
class Generator_RpGAN(nn.Module):
    def __init__(self):
        super(Generator_RpGAN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 128), # Input is State (Acceleration), matching LaTeX f(x), g(x)
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh()
        )
        # Outputs are Theta1 and Theta2 (Inverse Dynamics Parameters)
        # u = Theta1 + Theta2 * eta
        self.theta1_out = nn.Linear(128, 1)
        self.theta2_out = nn.Linear(128, 1)

        nn.init.constant_(self.theta1_out.bias, 0.0)
        nn.init.constant_(self.theta2_out.bias, 1.0)

    def forward(self, v_state):
        feat = self.hidden(v_state)
        theta1 = self.theta1_out(feat)
        theta2 = F.softplus(self.theta2_out(feat)) + 0.1 # Ensure non-singularity
        
        # Algebraic matching from Eq. 17: f = -Theta1/Theta2, g = 1/Theta2
        f = -theta1 / theta2
        g = 1.0 / theta2
        return f, g


class Discriminator_RpGAN(nn.Module):
    def __init__(self):
        super(Discriminator_RpGAN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, pair):
        return self.net(pair)


# ==========================================
# 3. Data Loading
# ==========================================
print("Loading dataset...")
try:
    df = pd.read_csv('offline_dataset_combined.csv')
    dataset_dict = {
        'X': {'v': df['v_dot_x'].values, 'v_dot': df['v_dot_x'].values, 'u': df['u_x'].values},
        'Y': {'v': df['v_dot_y'].values, 'v_dot': df['v_dot_y'].values, 'u': df['u_y'].values},
        'Z': {'v': df['v_dot_z'].values, 'v_dot': df['v_dot_z'].values, 'u': df['u_z'].values}
    }
except FileNotFoundError:
    print("Warning: Dataset not found. Using dummy data for testing.")
    dummy_data = np.random.randn(1000)
    dataset_dict = {
        'X': {'v': dummy_data, 'v_dot': dummy_data, 'u': dummy_data},
        'Y': {'v': dummy_data, 'v_dot': dummy_data, 'u': dummy_data},
        'Z': {'v': dummy_data, 'v_dot': dummy_data, 'u': dummy_data}
    }


# ==========================================
# 4. Core Training Function
# ==========================================
def train_axis_dynamics(axis_name, data, config):
    print(f"\n[{axis_name} Axis] Starting {config['gan_type']} training...")

    v_tensor = torch.FloatTensor(data['v']).unsqueeze(1).to(device)
    v_dot_tensor = torch.FloatTensor(data['v_dot']).unsqueeze(1).to(device)
    u_tensor = torch.FloatTensor(data['u']).unsqueeze(1).to(device)

    torch_dataset = Data.TensorDataset(u_tensor, v_dot_tensor, v_tensor)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=config['batch_size'], shuffle=True)

    generator = Generator_RpGAN().to(device)
    discriminator = Discriminator_RpGAN().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['lr_g'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['lr_d'])

    lsgan_criterion = nn.MSELoss() if config['gan_type'] == 'LSGAN' else None

    for epoch in range(config['epochs']):
        g_loss_avg = 0
        d_loss_avg = 0

        for b_u_real, b_vdot_real, b_v_real in loader:
            # Train Discriminator
            f, g = generator(b_v_real) # Input is State (Velocity)
            eta_fake = f + g * b_u_real
            eta_fake_det = eta_fake.detach()

            real_pair = torch.cat([b_vdot_real, b_vdot_real], dim=1)
            fake_pair_det = torch.cat([eta_fake_det, b_vdot_real], dim=1)

            d_real = discriminator(real_pair)
            d_fake_det = discriminator(fake_pair_det)

            if config['gan_type'] == 'LSGAN':
                d_loss_real = lsgan_criterion(d_real, torch.ones_like(d_real))
                d_loss_fake = lsgan_criterion(d_fake_det, torch.zeros_like(d_fake_det))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
            elif config['gan_type'] == 'GPGAN':
                gp = compute_gradient_penalty(discriminator, real_pair, fake_pair_det, device)
                d_loss = -torch.mean(d_real) + torch.mean(d_fake_det) + config['lambda_gp'] * gp

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            f_g, g_g = generator(b_v_real)
            eta_fake_g = f_g + g_g * b_u_real

            fake_pair_g = torch.cat([eta_fake_g, b_vdot_real], dim=1)
            d_fake_g = discriminator(fake_pair_g)

            if config['gan_type'] == 'LSGAN':
                gan_loss = lsgan_criterion(d_fake_g, torch.ones_like(d_fake_g))
            elif config['gan_type'] == 'GPGAN':
                gan_loss = -torch.mean(d_fake_g)

            anchor_loss = F.mse_loss(eta_fake_g, b_vdot_real)
            g_loss = gan_loss + config['lambda_anchor'] * anchor_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_loss_avg += d_loss.item()
            g_loss_avg += g_loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{config['epochs']}] | D Loss: {d_loss_avg / len(loader):.4f} | G Loss: {g_loss_avg / len(loader):.4f}")

    print(f"[{axis_name} Axis] Training completed.")
    return generator


# ==========================================
# 5. Execute Training
# ==========================================
trained_models = {}
print("\n==========================================")
print(f"Starting training process with {CONFIG['gan_type']}")
for axis in ['X', 'Y', 'Z']:
    G_model = train_axis_dynamics(axis, dataset_dict[axis], config=CONFIG)
    trained_models[axis] = G_model

print("\nAll models trained successfully.")


# ==========================================
# 6. Export to TorchScript (for C++ LibTorch)
# ==========================================
print("\n==========================================")
print("Exporting models to TorchScript (.pt)...")

for axis, model in trained_models.items():
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input for tracing
    dummy_input = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    
    # Trace model to generate static graph
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # Save exported model
    save_path = f"generator_prior_{axis}.pt"
    traced_model.save(save_path)
    
    print(f"[{axis} Axis] Exported to: {save_path}")

print("\nExport finished. Ready for C++ LibTorch inference.")