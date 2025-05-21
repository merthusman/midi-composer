import os
import sys
import time
import win32event
import win32api

def force_cleanup():
    try:
        # Farklı mutex isimlerini dene
        mutex_names = [
            "Global\\MIDI_Composer_Single_Instance_Mutex",
            "Local\\MIDI_Composer_Single_Instance_Mutex",
            "MIDI_Composer_Single_Instance_Mutex"
        ]
        
        cleaned = False
        
        for mutex_name in mutex_names:
            try:
                # MUTEX_ALL_ACCESS için 0x1F0001 kullanılıyor
                mutex = win32event.OpenMutex(0x1F0001, False, mutex_name)
                if mutex:
                    win32api.CloseHandle(mutex)
                    print(f"Mutex kapatıldı: {mutex_name}")
                    cleaned = True
            except Exception as e:
                print(f"Mutex kapatılırken hata ({mutex_name}): {e}")
        
        return cleaned
        
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        return False

def main():
    print("MIDI Composer Temizleme Aracı")
    print("=============================")
    
    print("\n1. Mutex temizleme işlemi başlatılıyor...")
    
    # Mutex temizleme işlemi
    if force_cleanup():
        print("\n✅ Mutex temizleme işlemi tamamlandı.")
    else:
        print("\nℹ️ Temizlenecek mutex bulunamadı veya zaten temiz.")
    
    # Kullanıcıya bilgi ver
    print("\nYapılması gerekenler:")
    print("1. Eğer arka planda çalışan bir uygulama varsa, görev yöneticisinden kapatın.")
    print("2. Uygulamayı yeniden başlatın.")
    print("\nEğer sorun devam ederse, lütfen bilgisayarınızı yeniden başlatın.")
    
    input("\nÇıkmak için Enter tuşuna basın...")

if __name__ == "__main__":
    main()
