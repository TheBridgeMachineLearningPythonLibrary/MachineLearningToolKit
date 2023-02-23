from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import time
import io
from PIL import Image
from mltoolkit.machine_learning import image_scrap
import shutil


def test_image_scrap():
    url = 'https://www.google.com/search?q=perros+bonitos&tbm=isch&ved=2ahUKEwiCpOG3z6n9AhVFV6QEHY7KBa0Q2-cCegQIABAA&oq=perros+bonitos&gs_lcp=CgNpbWcQAzIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBAgAEB4yBAgAEB4yBAgAEB5QwAlY6hFg6hJoAHAAeACAAYgBiAHJBpIBAzcuMpgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=YUz2Y8LvFMWukdUPjpWX6Ao&bih=849&biw=1600&rlz=1C5CHFA_enCA951CA951'
    n = 5
    image_scrap(url,n)

    download_dir = './my_images'
    
    assert os.path.exists(download_dir)
    assert len(os.listdir(download_dir)) == n

    shutil.rmtree(download_dir) 
