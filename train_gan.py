import torch
import torch.nn as nn
import torch.optim as optim
import time 
from tqdm import tqdm
from IPython import display
from utils import generate_images
import matplotlib.pyplot as plt
bce = nn.BCEWithLogitsLoss()

def discriminator_loss(real_output, generated_output):
    # Real images should be classified as 1 (real)
    real_loss = bce(real_output, torch.ones_like(real_output))

    # Generated (fake) images should be classified as 0 (fake)
    generated_loss = bce(generated_output, torch.zeros_like(generated_output))

    total_loss = real_loss + generated_loss
    
    return total_loss


def generator_loss(generated_output):
    
    return bce(generated_output, torch.ones_like(generated_output))



def train_step(images, 
               labels, 
               gen_loss_log, 
               disc_loss_log, 
               generator, 
               discriminator, 
               generator_optimizer, 
               discriminator_optimizer, 
               device="cpu", 
               noise_dim=100):
    
    current_batch_size = labels.shape[0]
    noise = torch.randn(current_batch_size, noise_dim, device=device)
    images, labels = images.to(device), labels.to(device)
    
    generator.train()
    discriminator.train()
    
    # Train Discriminator
    discriminator_optimizer.zero_grad()
    
    # Forward pass for real images
    real_output = discriminator(images, labels)
    
    # Forward pass for fake images
    generated_images = generator(noise, labels)
    generated_output = discriminator(generated_images.detach(), labels)
    
    # Compute discriminator loss
    disc_loss = discriminator_loss(real_output, generated_output)
    disc_loss_log.append(disc_loss.item())
    
    # Backpropagation for discriminator
    disc_loss.backward()
    discriminator_optimizer.step()
    
    # Train Generator
    generator_optimizer.zero_grad()
    
    # Forward pass through discriminator with generated images (without detach())
    generated_output = discriminator(generated_images, labels)
    
    # Compute generator loss
    gen_loss = generator_loss(generated_output)
    gen_loss_log.append(gen_loss.item())
    
    # Backpropagation for generator
    gen_loss.backward()
    generator_optimizer.step()
    
    return gen_loss.item(), disc_loss.item()
    
def train(dataloader, 
          generator, 
          discriminator, 
          gen_optimizer, 
          disc_optimizer,  
          epochs = 5, 
          device = 'cpu', 
          gen_loss_log = [], 
          disc_loss_log = [],
          test_noise = None,
          test_input_labels = None):    
    generator.to(device)
    discriminator.to(device)
    
    for epoch in range(epochs):
        start = time.time()

        gen_loss = 0
        disc_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                # Run the training step
                gen_loss, disc_loss = train_step(images, 
                                                labels, 
                                                gen_loss_log=gen_loss_log,
                                                disc_loss_log=disc_loss_log,
                                                generator=generator,
                                                discriminator=discriminator, 
                                                generator_optimizer=gen_optimizer,
                                                discriminator_optimizer=disc_optimizer,
                                                device = device)

                # Update the description in place after each batch
                pbar.set_postfix(Gen_Loss=gen_loss, Disc_Loss=disc_loss)

            
        
        if test_noise is not None and test_input_labels is not None:
            generate_images(model = generator, 
                                     epoch = epoch + 1,
                                     test_input_noise = test_noise, 
                                     test_input_labels = test_input_labels,
                                     device = device)
        
        print(f'Time taken for epoch {epoch + 1} is {time.time() - start:.2f} sec')
