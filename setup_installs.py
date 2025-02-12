#!/usr/bin/env python
# coding: utf-8

# # Welcome to UTMIST AI2!
# 
# [Technical Guide Notebook](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ)
# 
# [Introductory RL Notebook](https://colab.research.google.com/drive/1JRQFLU5jkMrIJ5cWs3xKEO0e9QKuE0Hi#scrollTo=9UCawVuAI3k0)
# 
# [Discord Server](https://discord.com/invite/TTGB62BE9U)
# 
# The link to the **LATEST VERSION** of this Colab will always be [here](https://docs.google.com/document/d/1SvlgQSUMLoO3cNx26hzViZOYVPAGa3ECrUSvUyVKvR4/edit?usp=sharing).
# 
# Credits:
# - General Event Organization: Asad, Efe, Andrew, Matthew, Kaden
# - Notebook code: Kaden, Martin, Andrew
# - Notebook art/animations: EchoTecho, Andy
# - Website code: Zain, Sarva, Adam, Aina
# - Workshops: Jessica, Jingmin, Asad, Tyler, Wai Lim, Napasorn, Sara, San, Alden
# - Tournament Server: Ambrose, Doga, Steven
# - Technical guide + Conference brochure: Matthew, Caitlin, Lucie

# # PATCH: Run this cell first

# In[ ]:


# Delete assets.zip and /content/assets/
import shutil, gdown, os
if os.path.exists('assets'):
    shutil.rmtree('assets')
if os.path.exists('assets.zip'):
    os.remove('assets.zip')

# Redownload from Drive
data_path = "assets.zip"
print("Downloading assets.zip...")
url = "https://drive.google.com/file/d/1F2MJQ5enUPVtyi3s410PUuv8LiWr8qCz/view?usp=sharing"
gdown.download(url, output=data_path, fuzzy=True)


# Unzip
get_ipython().system('unzip -q "/content/$data_path"')

# Delete attacks.zip and /content/attacks/
if os.path.exists('attacks'):
    shutil.rmtree('attacks')
if os.path.exists('attacks.zip'):
    os.remove('attacks.zip')

# Redownload from Drive
data_path = "attacks.zip"
print("Downloading attacks.zip...")
url = "https://drive.google.com/file/d/1LAOL8sYCUfsCk3TEA3vvyJCLSl0EdwYB/view?usp=sharing"
gdown.download(url, output=data_path, fuzzy=True)


# Unzip
get_ipython().system('unzip -q "/content/$data_path"')


# # pip installs

# In[ ]:


import gc
gc.collect()


# In[ ]:


# Download the requirements.txt from Google Drive
import gdown, os
data_path = "requirements.txt"
if not os.path.isfile(data_path):
    print("Downloading requirements.txt...")
    url = "https://drive.google.com/file/d/1-4f6NGWtejcn6Q9wUETelVXMFWaA5X0D/view?usp=sharing"
    gdown.download(url, output=data_path, fuzzy=True)


# In[ ]:


# Malachite and RL requirements
#!pip install torch==2.4.1 gymnasium pygame==2.6.1 pymunk==6.2.1 scikit-image scikit-video sympy==1.5.1 stable_baselines3 sb3-contrib
#!pip install memory_profiler==0.61.0
#!pip install torch==2.4.1 triton==3.0.0 gymnasium pygame==2.6.1 pymunk==6.2.1 scikit-image scikit-video sympy==1.5.1 stable_baselines3 sb3-contrib jupyter gdown opencv-python

#!pip freeze > /content/requirements_v0.txt
get_ipython().system('pip install -r requirements.txt')


# # Competition Code (DO NOT EDIT)
# Please **run this cell** to set your Jupyter Notebook up with all necessary code. Then feel free to move to the `SUBMISSION` sections of this notebook for further instruction.
# 
# Note: This cell may take time to run, as it installs the necessary modules then imports them.
# 
# ## Summary of content:
# 
# These cells contain our custom Multi-Agent Reinforcement Learning (MARL) Solution, Malachite, alongside an implementation of a 1v1 platform fighter we've titled Warehouse Brawl. You may look through these cells to get a better sense for the dynamics and functionality of the enviroment, as wel as the various pip installs and modules available for use in your own code.

# ## Malachite (DO NOT MODIFY UNLESS YOU KNOW WHAT YOU'RE DOING)
# The following cells store some code for our custom Multi-Agent Reinforcement Learning (MARL) Solution, called Malachite. It extends some Stable-Baselines 3 functionality to Multi-Agent systems in the context of the AI2 Tournament.
# 
# You would only want to modify this if you want to add custom rewards and are dissatisfied with the current flexible rewards system. That's ok! But note that this default environment is what will be used in the tournament, and you will **NOT** have access to any additional data or modifications you may choose to make here.

