from distutils.core import setup


install_requires = [
    "sounddevice==0.4.4",
    "soundfile==0.10.3.post1",
    "librosa==0.8.1",
    "sox==1.4.1",
    "numpy",
    "ffmpy==0.3.0",
    "maglevapi"
]


setup(
    name="pyaudiomixer",
    packages=["pyaudio_mixer"],
    version="0.1",
    license="MIT",
    description="Audio mixing simplified.",
    author="Philippe Mathew",
    author_email="philmattdev@gmail.com",
    url="",
    download_url="",
    keywords=["audio", "mixer", "mixing", "tool"],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
    ]
)

