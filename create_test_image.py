# Test resmi oluşturmak için basit bir script
from PIL import Image, ImageDraw, ImageFont
import os

# Resim boyutları
width, height = 1920, 1080

# Yeni bir resim oluştur
image = Image.new('RGB', (width, height), color=(20, 22, 36))

# Çizim nesnesi oluştur
draw = ImageDraw.Draw(image)

# Basit bir gradyan arka plan oluştur
for y in range(height):
    # Y pozisyonuna göre renk değişimi
    r = int(20 + (y / height) * 40)
    g = int(22 + (y / height) * 60)
    b = int(36 + (y / height) * 100)
    
    # Yatay çizgi çiz
    draw.line([(0, y), (width, y)], fill=(r, g, b))

# Metin ekle
try:
    # Yazı tipi yüklemeye çalış
    font = ImageFont.truetype("arial.ttf", 120)
except:
    # Yazı tipi yüklenemezse varsayılan yazı tipini kullan
    font = ImageFont.load_default()

# Metin ekle
draw.text((width//2, height//2), "MIDI COMPOSER", fill=(255, 255, 255), font=font, anchor="mm")

# Resmi kaydet
output_path = os.path.join("resources", "images", "background.jpg")
image.save(output_path, quality=95)

print(f"Test resmi başarıyla oluşturuldu: {output_path}")
