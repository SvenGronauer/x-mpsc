import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="x_mpsc",  # this is the name displayed in 'pip list'
    description="X-model predictive safety certification.",
    install_requires=[
        'casadi',
        'cvxpy',
        'dill==0.3.7',
        'gymnasium==0.29.1',
        'matplotlib',
        'mpi4py',
        'numpy==1.23.5',
        'ffmpeg-python',
        'torch',
        'pygame',
        'pandas',
        'psutil',
        'scipy',
        'tensorboard',
        'tqdm',
        'joblib'  # required for running LBPO
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
