from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Union
import os
import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime, timedelta
import io
import base64
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging  # Loglama modülünü ekle
import sys # sys modülünü ekle (isteğe bağlı, handler için)
import functools # lru_cache için eklendi
import time # Middleware için eklendi
import uuid # uuid eklendi

# --- Loglama Yapılandırması ---
# logging.basicConfig satırlarını kaldırıyoruz.

# Loglama yapılandırmasını sözlük olarak tanımla
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Mevcut logger'ları (uvicorn dahil) devre dışı bırakma
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Konsola yazdır
        },
    },
    "loggers": {
        "": {  # Kök logger (tüm logger'ları kapsar)
            "handlers": ["default"],
            "level": "DEBUG",  # Seviyeyi DEBUG yapalım (daha sonra INFO'ya çekebilirsin)
            "propagate": True # Olayların üst logger'lara gitmesine izin ver
        },
        "uvicorn.error": {
            "level": "INFO", # Uvicorn hatalarını INFO seviyesinde tut
             "handlers": ["default"],
             "propagate": False # Kök logger'a tekrar gitmesin
        },
         "uvicorn.access": {
             "handlers": [], # Erişim loglarını kendi formatımızla yazdırmak için devre dışı bırakabiliriz
             "propagate": False # ya da middleware loglarımız yeterliyse tamamen kapatabiliriz. Şimdilik kapalı kalsın.
         }
    }
}

logger = logging.getLogger(__name__) # Logger'ı yine alıyoruz, yapılandırma Uvicorn tarafından uygulanacak
# -----------------------------

# --- Grafik Depolama ---
chart_storage: Dict[str, Dict[str, Any]] = {} # Grafik verilerini saklamak için dictionary
CHART_EXPIRY = timedelta(hours=24) # Grafiklerin saklanma süresi (örneğin 24 saat)
# -----------------------

# Uyarıları gösterme
warnings.filterwarnings("ignore")

###############################################################################
# 1) FastAPI Uygulaması ve API Anahtarı Güvenlik Mekanizması
###############################################################################
app = FastAPI(
    title="Borsa Teknik Analiz API",
    description="Hisse senetleri için teknik analiz araçları sunan API",
    version="1.0.0"
)

# --- EKLEMEN GEREKEN SİHİRLİ KISIM (CORS AYARI) ---
# Bu kısım FlutterFlow'un ve tarayıcıların API'ye erişmesine izin verir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm sitelere kapıyı açar.
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST her şeye izin verir.
    allow_headers=["*"],  # API Key vb. başlıklara izin verir.
)
# ----------------------------------------------------

# --- Middleware for Request Logging ---
@app.middleware("http")
async def log_requests(request, call_next):
    """Gelen istekleri ve işlem sürelerini loglar."""
    idem = f"{request.client.host}:{request.client.port}" # Basit bir istek kimliği
    logger.info(f"rid={idem} Gelen istek: {request.method} {request.url.path}")
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000 # Milisaniye cinsinden
    formatted_process_time = f"{process_time:.2f}ms"
    logger.info(f"rid={idem} İstek tamamlandı: {response.status_code}, Süre: {formatted_process_time}")
    
    return response

# API güvenliği için bir header belirliyoruz.
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY", "test_key")  # Örnek anahtar, canlıda değiştirin!

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Header'dan gelen anahtarın doğruluğunu kontrol eder."""
    if not api_key_header:
        logger.warning("API anahtarı başlıkta eksik.")
        raise HTTPException(
            status_code=403, detail="API anahtarı eksik"
        )
    if api_key_header != API_KEY:
        logger.warning(f"Geçersiz API anahtarı denemesi: {api_key_header[:5]}...") # Anahtarın tamamını loglama
        raise HTTPException(
            status_code=403, detail="API anahtarı geçersiz"
        )
    logger.info("API anahtarı başarıyla doğrulandı.")
    return api_key_header

###############################################################################
# 2) Teknik Analiz İçin Veri Modelleri
###############################################################################
class TechnicalAnalysisRequest(BaseModel):
    """
    Teknik analiz isteği parametreleri
    """
    symbol: str = Field(..., description="Hisse senedi sembolü (örn: AAPL, THYAO.IS)")
    start_date: str = Field(None, description="Başlangıç tarihi (YYYY-MM-DD)")
    end_date: str = Field(None, description="Bitiş tarihi (YYYY-MM-DD)")
    include_charts: bool = Field(False, description="Grafik verilerini dahil et")

class TechnicalAnalysisResponse(BaseModel):
    """
    Teknik analiz sonuç modeli
    """
    symbol: str
    period: str
    data: Dict[str, Any]
    indicators: Dict[str, Any]
    fibonacci_levels: Dict[str, float]
    charts: Optional[Dict[str, str]] = None

###############################################################################
# 3) Teknik Analiz Fonksiyonları 
###############################################################################
def turkce_ay_ismi(tarih):
    """
    Datetime nesnesinden Türkçe ay ismini döndürür
    """
    aylar = {
        1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan", 5: "Mayıs", 6: "Haziran",
        7: "Temmuz", 8: "Ağustos", 9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
    }
    return aylar[tarih.month]

def get_stock_data(sembol, baslangic_tarih, bitis_tarih):
    """
    Yahoo Finance'dan hisse senedi verilerini çeker (Önbelleğe alınmaz, ana fonksiyon önbelleğe alınır)
    """
    logger.debug(f"'{sembol}' için veri çekiliyor (fonksiyon içi): {baslangic_tarih} - {bitis_tarih}")
    
    # Türk hissesi kontrolü - sembol zaten .IS içermiyorsa ekle
    if sembol.upper().endswith(".IS") is False and len(sembol) <= 5:  # Türk hisseleri genellikle kısa kodlar
        yahoo_sembol = f"{sembol.upper()}.IS"
        logger.debug(f"Türk hissesi tespit edildi, sembol Yahoo Finance formatına dönüştürüldü: {yahoo_sembol}")
    else:
        yahoo_sembol = sembol
    
    try:
        # yfinance'ın kendi loglarını azaltmak için progress=False ve uyarıyı susturmak için auto_adjust=True
        veri = yf.download(
            yahoo_sembol, 
            start=baslangic_tarih, 
            end=bitis_tarih, 
            progress=False, 
            auto_adjust=True # Bu uyarıyı susturur
        ) 
        
        if veri.empty:
            logger.warning(f"'{sembol}' için belirtilen aralıkta veri bulunamadı.")
            return None 

        # MultiIndex sütunlar sorunu düzeltme (Eğer tek sembol ise)
        if isinstance(veri.columns, pd.MultiIndex) and len(veri.columns.levels) > 1:
            if veri.columns.get_level_values(1).nunique() == 1:
                logger.debug(f"'{sembol}' için MultiIndex sütunlar düzeltiliyor.")
                veri.columns = veri.columns.get_level_values(0)
        
        logger.debug(f"'{sembol}' için {len(veri)} satır veri başarıyla çekildi.")
        return veri
    except Exception as e:
        logger.error(f"'{sembol}' için veri çekme sırasında hata oluştu: {e}", exc_info=True)
        return None 

def hesapla_ma(veri, periyot=20, tur="SMA"):
    """
    Hareketli Ortalama hesaplar (SMA, EMA)
    """
    if tur == "SMA":
        veri[f'SMA_{periyot}'] = veri['Close'].rolling(window=periyot).mean()
    elif tur == "EMA":
        veri[f'EMA_{periyot}'] = veri['Close'].ewm(span=periyot, adjust=False).mean()
    
    return veri

def hesapla_rsi(veri, periyot=14):
    """
    RSI (Göreceli Güç Endeksi) hesaplar
    """
    delta = veri['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=periyot).mean()
    avg_loss = loss.rolling(window=periyot).mean()
    
    # İlk RSI hesaplaması
    rs = avg_gain / avg_loss
    veri['RSI'] = 100 - (100 / (1 + rs))
    
    return veri

def hesapla_bollinger_bantlari(veri, periyot=20, stddev=2):
    """
    Bollinger Bantlarını hesaplar
    """
    veri['BB_Mid'] = veri['Close'].rolling(window=periyot).mean()
    veri['BB_Std'] = veri['Close'].rolling(window=periyot).std()
    veri['BB_Upper'] = veri['BB_Mid'] + (veri['BB_Std'] * stddev)
    veri['BB_Lower'] = veri['BB_Mid'] - (veri['BB_Std'] * stddev)
    
    return veri

def hesapla_macd(veri, fast=12, slow=26, signal=9):
    """
    MACD (Moving Average Convergence Divergence) hesaplar
    """
    # MACD Line = 12-günlük EMA - 26-günlük EMA
    veri['EMA_fast'] = veri['Close'].ewm(span=fast, adjust=False).mean()
    veri['EMA_slow'] = veri['Close'].ewm(span=slow, adjust=False).mean()
    veri['MACD'] = veri['EMA_fast'] - veri['EMA_slow']
    
    # Signal Line = 9-günlük EMA of MACD Line
    veri['MACD_Signal'] = veri['MACD'].ewm(span=signal, adjust=False).mean()
    
    # Histogram = MACD Line - Signal Line
    veri['MACD_Histogram'] = veri['MACD'] - veri['MACD_Signal']
    
    return veri

def hesapla_fibonacci_seviyeleri(veri):
    """
    Fibonacci düzeltme seviyelerini hesaplar
    """
    yuksek_nokta = float(veri['High'].max())
    dusuk_nokta = float(veri['Low'].min())
    
    fark = yuksek_nokta - dusuk_nokta
    
    # Fibonacci seviyeleri
    fib_seviyeler = {
        '0.0': float(dusuk_nokta),
        '0.236': float(dusuk_nokta + 0.236 * fark),
        '0.382': float(dusuk_nokta + 0.382 * fark),
        '0.5': float(dusuk_nokta + 0.5 * fark),
        '0.618': float(dusuk_nokta + 0.618 * fark),
        '0.786': float(dusuk_nokta + 0.786 * fark),
        '1.0': float(yuksek_nokta)
    }
    
    return fib_seviyeler

def ciz_fiyat_grafigi(veri, sembol):
    """
    Plotly ile fiyat ve hareketli ortalama grafiği çizer
    """
    fig = go.Figure()
    
    # Mum çubuğu grafiği
    fig.add_trace(
        go.Candlestick(
            x=veri.index,
            open=veri['Open'],
            high=veri['High'],
            low=veri['Low'],
            close=veri['Close'],
            name='Fiyat',
            increasing_line_color='#26A69A',  # Yeşil - yükseliş
            decreasing_line_color='#EF5350',  # Kırmızı - düşüş
        )
    )
    
    # Hareketli Ortalamalar
    ma_colors = ['#2962FF', '#FF6D00', '#E91E63', '#4CAF50']
    color_index = 0
    
    for ma_type in ['SMA', 'EMA']:
        for periyot in [20, 50]:
            col_name = f'{ma_type}_{periyot}'
            if col_name in veri.columns:
                fig.add_trace(
                    go.Scatter(
                        x=veri.index,
                        y=veri[col_name],
                        name=f'{ma_type} {periyot}',
                        line=dict(color=ma_colors[color_index], width=1.5)
                    )
                )
                color_index = (color_index + 1) % len(ma_colors)
    
    # Grafik düzeni
    fig.update_layout(
        title=f'{sembol} Fiyat ve Hareketli Ortalamalar',
        xaxis_title='Tarih',
        yaxis_title='Fiyat',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=600
    )
    
    # Tarihleri 6 parçaya böl (timedelta ile çalışması için)
    try:
        tarih_araligi = max((veri.index[-1] - veri.index[0]).days, 1)
        nticks = 6  # Görünecek tarih sayısı
        dtick = max(1, tarih_araligi // nticks)  # En az 1 günlük aralık olsun
        
        # X ekseni tarih formatı - Türkçe format için güncellendi
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hafta sonlarını gizle
            ],
            tickformat="%d-%m-%Y",  # Gün-Ay-Yıl formatında göster
            tickangle=-45,  # Okunaklı olması için açı ver
            tickmode="auto",         # Otomatik tick modu
            nticks=nticks,           # İstenen tick sayısı
        )
    except Exception as e:
        # Hata olursa varsayılan formatta göster
        logger.error(f"Tarih ekseni formatlanırken hata: {e}")
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            tickformat="%d-%m-%Y",
            tickangle=-45
        )
    
    return fig

def ciz_bollinger_fibonacci_grafigi(veri, sembol, fib_seviyeler):
    """
    Plotly ile Bollinger Bantları ve Fibonacci seviyeleri grafiği çizer
    """
    fig = go.Figure()
    
    # Mum çubuğu grafiği
    fig.add_trace(
        go.Candlestick(
            x=veri.index,
            open=veri['Open'],
            high=veri['High'],
            low=veri['Low'],
            close=veri['Close'],
            name='Fiyat',
            increasing_line_color='#26A69A',  # Yeşil - yükseliş
            decreasing_line_color='#EF5350',  # Kırmızı - düşüş
        )
    )
    
    # Bollinger Bantları
    if 'BB_Upper' in veri.columns:
        fig.add_trace(
            go.Scatter(
                x=veri.index,
                y=veri['BB_Upper'],
                name='BB Üst',
                line=dict(color='#1E88E5', width=1, dash='dash')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=veri.index,
                y=veri['BB_Lower'],
                name='BB Alt',
                line=dict(color='#1E88E5', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(30, 136, 229, 0.1)'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=veri.index,
                y=veri['BB_Mid'],
                name='BB Orta',
                line=dict(color='#1E88E5', width=1.5)
            )
        )
    
    # Fibonacci Seviyeleri
    if fib_seviyeler:
        colors = ['#f44336', '#ff9800', '#ffc107', '#4caf50', '#2196f3', '#3f51b5', '#9c27b0']
        color_idx = 0
        
        # Önce grafik eksenlerini ayarla
        y_min = min(veri['Low'].min(), min(fib_seviyeler.values()))
        y_max = max(veri['High'].max(), max(fib_seviyeler.values()))
        fig.update_yaxes(range=[y_min * 0.98, y_max * 1.02])  # %2 marj ekle
        
        for seviye, deger in fib_seviyeler.items():
            # Fib seviyesinin açıklaması
            if seviye == '0.0':
                label = 'Fibonacci 0.0 (Dip)'
            elif seviye == '1.0':
                label = 'Fibonacci 1.0 (Tepe)'
            else:
                label = f'Fibonacci {seviye}'
            
            # Yatay çizgi ve etiketi ekle
            fig.add_shape(
                type="line",
                x0=veri.index[0],
                y0=float(deger),
                x1=veri.index[-1],
                y1=float(deger),
                line=dict(
                    color=colors[color_idx],
                    width=1.5,
                    dash="dash",
                ),
            )
            
            # Metin ekle - sağ tarafta, arkaplanı beyaz, belirgin
            fig.add_annotation(
                x=veri.index[-1],
                y=float(deger),
                text=f"{label}: {float(deger):.2f}",
                showarrow=False,
                xshift=10,  # Biraz sağa kaydır
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor=colors[color_idx],
                borderwidth=1,
                borderpad=4,
                font=dict(
                    size=10,
                    color=colors[color_idx]
                ),
                align="left"
            )
            
            color_idx = (color_idx + 1) % len(colors)
    
    # Grafik düzeni
    fig.update_layout(
        title=f'{sembol} Bollinger Bantları ve Fibonacci Seviyeleri',
        xaxis_title='Tarih',
        yaxis_title='Fiyat',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=600
    )
    
    # Tarihleri 6 parçaya böl (timedelta ile çalışması için)
    try:
        tarih_araligi = max((veri.index[-1] - veri.index[0]).days, 1)
        nticks = 6  # Görünecek tarih sayısı
        
        # X ekseni tarih formatı - Türkçe format için güncellendi
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hafta sonlarını gizle
            ],
            tickformat="%d-%m-%Y",  # Gün-Ay-Yıl formatında göster
            tickangle=-45,  # Okunaklı olması için açı ver
            tickmode="auto",         # Otomatik tick modu
            nticks=nticks,           # İstenen tick sayısı
        )
    except Exception as e:
        # Hata olursa varsayılan formatta göster
        logger.error(f"Tarih ekseni formatlanırken hata: {e}")
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            tickformat="%d-%m-%Y",
            tickangle=-45
        )
    
    return fig

def ciz_teknik_gostergeler_grafigi(veri, sembol):
    """
    Plotly ile RSI ve MACD göstergeleri grafiği çizer
    """
    # Subplot oluştur - fiyat ve teknik göstergeler
    row_heights = [0.5, 0.25, 0.25]
    specs = [[{"type": "candlestick"}], [{"type": "scatter"}], [{"type": "scatter"}]]
    rows = 3
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=(f"{sembol} Fiyat", "RSI", "MACD")
    )
    
    # 1. Fiyat Grafiği
    fig.add_trace(
        go.Candlestick(
            x=veri.index,
            open=veri['Open'],
            high=veri['High'],
            low=veri['Low'],
            close=veri['Close'],
            name='Fiyat',
            increasing_line_color='#26A69A',  # Yeşil - yükseliş
            decreasing_line_color='#EF5350',  # Kırmızı - düşüş
        ),
        row=1, col=1
    )
    
    # 2. RSI Grafiği
    if 'RSI' in veri.columns:
        fig.add_trace(
            go.Scatter(
                x=veri.index,
                y=veri['RSI'],
                name='RSI',
                line=dict(color='purple', width=1.5)
            ),
            row=2, col=1
        )
        
        # RSI aşırı alım/satım seviyeleri
        fig.add_hline(
            y=70,
            line_width=1,
            line_dash="dash",
            line_color="red",
            row=2, col=1,
            annotation_text="Aşırı Alım (70)",
            annotation_position="right"
        )
        
        fig.add_hline(
            y=30,
            line_width=1,
            line_dash="dash",
            line_color="green",
            row=2, col=1,
            annotation_text="Aşırı Satım (30)",
            annotation_position="right"
        )
    
    # 3. MACD Grafiği
    if all(x in veri.columns for x in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
        fig.add_trace(
            go.Scatter(
                x=veri.index,
                y=veri['MACD'],
                name='MACD',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=veri.index,
                y=veri['MACD_Signal'],
                name='Sinyal',
                line=dict(color='red', width=1.5)
            ),
            row=3, col=1
        )
        
        # MACD Histogram renklerini belirle
        colors = ['#26A69A' if v >= 0 else '#EF5350' for v in veri['MACD_Histogram']]
        
        # MACD Histogram çubuklarını ekle
        fig.add_trace(
            go.Bar(
                x=veri.index,
                y=veri['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.5
            ),
            row=3, col=1
        )
    
    # Grafik düzeni
    fig.update_layout(
        title=f'{sembol} Teknik Göstergeler',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=900
    )
    
    # Tarihleri 6 parçaya böl (timedelta ile çalışması için)
    try:
        tarih_araligi = max((veri.index[-1] - veri.index[0]).days, 1)
        nticks = 6  # Görünecek tarih sayısı
        
        # X ekseni tarih formatı - Türkçe format için güncellendi (tüm alt grafikler için)
        for i in range(1, rows+1):
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # Hafta sonlarını gizle
                ],
                tickformat="%d-%m-%Y",  # Gün-Ay-Yıl formatında göster
                tickangle=-45,  # Okunaklı olması için açı ver
                tickmode="auto",         # Otomatik tick modu
                nticks=nticks,           # İstenen tick sayısı
                row=i, col=1
            )
    except Exception as e:
        # Hata olursa varsayılan formatta göster
        logger.error(f"Tarih ekseni formatlanırken hata: {e}")
        for i in range(1, rows+1):
            fig.update_xaxes(
                rangebreaks=[dict(bounds=["sat", "mon"])],
                tickformat="%d-%m-%Y",
                tickangle=-45,
                row=i, col=1
            )
    
    return fig

def figur_to_base64(fig):
    """Plotly figürünü base64 kodlu içeriğe dönüştürür. PNG başarısız olursa HTML'e düşer.
    Returns: (base64_str, media_type)
    """
    try:
        img_bytes = fig.to_image(format="png")
        return base64.b64encode(img_bytes).decode('utf-8'), "image/png"
    except Exception as e:
        logger.warning(f"PNG üretimi başarısız, HTML'e düşülüyor: {e}")
        try:
            html = fig.to_html(full_html=True, include_plotlyjs='cdn')
            return base64.b64encode(html.encode('utf-8')).decode('utf-8'), "text/html; charset=utf-8"
        except Exception as e2:
            logger.error(f"Grafik HTML üretimi de başarısız: {e2}")
            raise

# Ana analiz fonksiyonunu önbelleğe alalım
@functools.lru_cache(maxsize=256)
def ana_teknik_analiz(sembol, baslangic_tarih=None, bitis_tarih=None, include_charts=False):
    """
    Verilen sembol için tüm teknik analiz hesaplamalarını yapar.
    Sonuçlar önbelleğe alınır.
    """
    # Önbellek bilgisini logla (Bu kısım lru_cache'in kendi istatistikleri ile daha iyi yapılabilir ama basitlik için böyle)
    # Gerçek cache hit/miss loglaması için daha gelişmiş bir cache mekanizması veya wrapper gerekebilir.
    # Şimdilik sadece fonksiyonun çağrıldığını loglayalım. Cache'den gelirse bu log tekrar görünmez.
    logger.info(f"Önbellekte yok veya süresi dolmuş: '{sembol}' için analiz hesaplanıyor...")
    logger.debug(f"Parametreler: Başlangıç={baslangic_tarih}, Bitiş={bitis_tarih}, Grafik={include_charts}")

    # Varsayılan tarihler
    effective_start_date = baslangic_tarih if baslangic_tarih else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    effective_end_date = bitis_tarih if bitis_tarih else datetime.now().strftime('%Y-%m-%d')
    
    # Veri al
    logger.debug(f"'{sembol}' için get_stock_data çağrılıyor...")
    veri = get_stock_data(sembol, effective_start_date, effective_end_date)
    
    # Veri alınamazsa veya boşsa
    if veri is None or veri.empty:
        logger.error(f"'{sembol}' için teknik analiz yapılamadı, veri alınamadı veya boş.")
        # Önbelleğe alınacak None değerleri döndür
        return None, None, None, None, None 
    
    logger.debug(f"'{sembol}' için teknik göstergeler hesaplanıyor...")
    # Teknik göstergeler hesapla
    try:
        logger.debug("Hesaplanıyor: MA(20, SMA), MA(50, SMA), MA(20, EMA), MA(50, EMA)")
        veri = hesapla_ma(veri, 20, "SMA")
        veri = hesapla_ma(veri, 50, "SMA")
        veri = hesapla_ma(veri, 20, "EMA")
        veri = hesapla_ma(veri, 50, "EMA")
        logger.debug("Hesaplanıyor: RSI(14)")
        veri = hesapla_rsi(veri)
        logger.debug("Hesaplanıyor: Bollinger Bantları(20, 2)")
        veri = hesapla_bollinger_bantlari(veri)
        logger.debug("Hesaplanıyor: MACD(12, 26, 9)")
        veri = hesapla_macd(veri)
        logger.debug(f"'{sembol}' için temel göstergeler hesaplandı.")
        
        logger.debug("Hesaplanıyor: Fibonacci Seviyeleri")
        fib_seviyeler = hesapla_fibonacci_seviyeleri(veri)
        logger.debug(f"'{sembol}' için Fibonacci seviyeleri hesaplandı.")
        
    except Exception as e:
        logger.error(f"'{sembol}' için teknik gösterge hesaplama sırasında hata: {e}", exc_info=True)
        return None, None, None, None, None

    # Grafikler (opsiyonel) - Artık ID'leri saklayacağız
    chart_ids = {} # Base64 yerine ID'leri tutacak dictionary
    if include_charts:
        logger.debug(f"'{sembol}' için grafikler oluşturuluyor ve saklanıyor...")
        chart_start_time = time.time()
        try:
            # --- Fiyat Grafiği ---
            logger.debug("Oluşturuluyor ve saklanıyor: Fiyat Grafiği")
            fiyat_grafik = ciz_fiyat_grafigi(veri, sembol)
            fiyat_base64, fiyat_media = figur_to_base64(fiyat_grafik)
            fiyat_chart_id = str(uuid.uuid4())[:8] # Kısa ID oluştur
            chart_storage[fiyat_chart_id] = {
                "data": fiyat_base64,
                "media_type": fiyat_media,
                "expires_at": datetime.now() + CHART_EXPIRY
            }
            chart_ids['price_chart_id'] = fiyat_chart_id # ID'yi sakla
            logger.debug(f"Fiyat grafiği saklandı: chart_id={fiyat_chart_id}")

            # --- Bollinger & Fibonacci Grafiği ---
            logger.debug("Oluşturuluyor ve saklanıyor: Bollinger & Fibonacci Grafiği")
            bollinger_fib_grafik = ciz_bollinger_fibonacci_grafigi(veri, sembol, fib_seviyeler)
            bollinger_fib_base64, bollinger_media = figur_to_base64(bollinger_fib_grafik)
            bollinger_fib_chart_id = str(uuid.uuid4())[:8]
            chart_storage[bollinger_fib_chart_id] = {
                "data": bollinger_fib_base64,
                "media_type": bollinger_media,
                "expires_at": datetime.now() + CHART_EXPIRY
            }
            chart_ids['bollinger_fib_chart_id'] = bollinger_fib_chart_id
            logger.debug(f"Bollinger/Fib grafiği saklandı: chart_id={bollinger_fib_chart_id}")

            # --- Teknik Göstergeler Grafiği ---
            logger.debug("Oluşturuluyor ve saklanıyor: Teknik Göstergeler Grafiği")
            teknik_grafik = ciz_teknik_gostergeler_grafigi(veri, sembol)
            teknik_base64, teknik_media = figur_to_base64(teknik_grafik)
            teknik_chart_id = str(uuid.uuid4())[:8]
            chart_storage[teknik_chart_id] = {
                "data": teknik_base64,
                "media_type": teknik_media,
                "expires_at": datetime.now() + CHART_EXPIRY
            }
            chart_ids['indicators_chart_id'] = teknik_chart_id
            logger.debug(f"Teknik göstergeler grafiği saklandı: chart_id={teknik_chart_id}")

            chart_process_time = (time.time() - chart_start_time) * 1000
            logger.debug(f"'{sembol}' için 3 grafik başarıyla oluşturuldu ve saklandı ({chart_process_time:.2f}ms).")
        except Exception as e:
            logger.error(f"'{sembol}' için grafik oluşturma/saklama sırasında hata: {e}", exc_info=True)
            chart_ids = {} # Hata durumunda ID listesini boşalt

    # Son 10 günlük veriler
    son_veriler = veri.tail(10)
    
    # Gösterge değerlerini hazırla
    logger.debug(f"'{sembol}' için son gösterge değerleri hazırlanıyor...")
    indicators = {}
    last_data = veri.iloc[-1]
    
    # RSI
    if 'RSI' in veri.columns and not pd.isna(last_data['RSI']):
        rsi_value = last_data['RSI']
        rsi_status = "Aşırı alım" if rsi_value > 70 else "Aşırı satım" if rsi_value < 30 else "Normal"
        indicators['rsi'] = {'value': float(rsi_value), 'status': rsi_status}
        logger.debug(f"Son RSI: {rsi_value:.2f} ({rsi_status})")
    else:
        indicators['rsi'] = {'value': None, 'status': 'Hesaplanamadı'}
        logger.debug("Son RSI hesaplanamadı.")

    # MACD
    if all(x in veri.columns for x in ['MACD', 'MACD_Signal']) and not pd.isna(last_data['MACD']) and not pd.isna(last_data['MACD_Signal']):
        macd_value = last_data['MACD']
        signal_value = last_data['MACD_Signal']
        histogram_value = last_data.get('MACD_Histogram') # .get() ile daha güvenli
        macd_status = "Yükseliş" if macd_value > signal_value else "Düşüş"
        indicators['macd'] = {
            'macd': float(macd_value),
            'signal': float(signal_value),
            'histogram': float(histogram_value) if histogram_value is not None and not pd.isna(histogram_value) else None,
            'status': macd_status
        }
        logger.debug(f"Son MACD: {macd_value:.2f}, Sinyal: {signal_value:.2f} ({macd_status})")
    else:
         indicators['macd'] = {'macd': None, 'signal': None, 'histogram': None, 'status': 'Hesaplanamadı'}
         logger.debug("Son MACD hesaplanamadı.")

    # Bollinger Bantları
    if all(x in veri.columns for x in ['BB_Mid', 'BB_Upper', 'BB_Lower']) and not pd.isna(last_data['BB_Mid']):
        bb_mid = last_data['BB_Mid']
        bb_upper = last_data['BB_Upper']
        bb_lower = last_data['BB_Lower']
        current_price = last_data['Close']
        bb_status = "Bantlar içinde"
        if not pd.isna(current_price):
             if current_price > bb_upper: bb_status = "Üst bandın üstünde (aşırı alım)"
             elif current_price < bb_lower: bb_status = "Alt bandın altında (aşırı satım)"
        indicators['bollinger'] = {
            'middle': float(bb_mid),
            'upper': float(bb_upper) if not pd.isna(bb_upper) else None,
            'lower': float(bb_lower) if not pd.isna(bb_lower) else None,
            'status': bb_status
        }
        logger.debug(f"Son Bollinger: Alt={bb_lower:.2f}, Orta={bb_mid:.2f}, Üst={bb_upper:.2f} ({bb_status})")
    else:
        indicators['bollinger'] = {'middle': None, 'upper': None, 'lower': None, 'status': 'Hesaplanamadı'}
        logger.debug("Son Bollinger Bantları hesaplanamadı.")

    # Hareketli Ortalamalar
    moving_averages = {}
    ma_log_parts = []
    for ma_type in ['SMA', 'EMA']:
        for period in [20, 50]:
            col_name = f'{ma_type}_{period}'
            if col_name in veri.columns and not pd.isna(last_data[col_name]):
                ma_value = last_data[col_name]
                moving_averages[col_name] = float(ma_value)
                ma_log_parts.append(f"{col_name}={ma_value:.2f}")
            else:
                 moving_averages[col_name] = None
    indicators['moving_averages'] = moving_averages
    logger.debug(f"Son Hareketli Ortalamalar: {', '.join(ma_log_parts)}")
    
    # Son fiyat ve değişim
    try:
        # Ensure there are at least two data points for pct_change
        if len(veri['Close']) > 1:
            change_pct = veri['Close'].pct_change().iloc[-1] * 100
            if pd.isna(change_pct): # Handle potential NaN if previous close was 0 or NaN
                 change_pct = 0.0
                 logger.warning(f"'{sembol}' için yüzde değişim NaN döndü, 0 olarak ayarlandı.")
        else:
            change_pct = 0.0 
            logger.warning(f"'{sembol}' için yüzde değişim hesaplanamadı (sadece 1 veri noktası).")
    except IndexError:
        change_pct = 0.0 
        logger.warning(f"'{sembol}' için yüzde değişim hesaplanırken IndexError.")
    except Exception as e:
        change_pct = 0.0
        logger.error(f"'{sembol}' için yüzde değişim hesaplanırken hata: {e}", exc_info=True)


    indicators['price'] = {
        'open': float(last_data['Open']) if not pd.isna(last_data['Open']) else None,
        'high': float(last_data['High']) if not pd.isna(last_data['High']) else None,
        'low': float(last_data['Low']) if not pd.isna(last_data['Low']) else None,
        'close': float(last_data['Close']) if not pd.isna(last_data['Close']) else None,
        'volume': float(last_data['Volume']) if 'Volume' in last_data and not pd.isna(last_data['Volume']) else None,
        'change': float(change_pct) # Zaten float veya 0.0
    }
    logger.debug(f"Son Fiyat: Kapanış={indicators['price']['close']:.2f}, Değişim={indicators['price']['change']:.2f}%")
    
    logger.info(f"'{sembol}' için analiz hesaplaması başarıyla tamamlandı.")
    # Önbelleğe alınacak değerleri döndür (charts yerine chart_ids)
    return veri, son_veriler, fib_seviyeler, indicators, chart_ids # charts yerine chart_ids döndürülüyor

###############################################################################
# 4) API Endpoint'leri
###############################################################################
@app.get("/")
async def root():
    """
    API hakkında temel bilgi veren ana endpoint
    """
    return {
        "name": "Borsa Teknik Analiz API",
        "version": "1.0.0",
        "description": "Hisse senetleri için teknik analiz araçları sunan API",
        "auth_header": f"Bu servisi kullanmak için {API_KEY_NAME} header'ı ile geçerli bir API anahtarı yollayın.",
        "endpoints": {
            "POST /api/teknik-analiz": "Hisse senedi için detaylı teknik analiz yapar.",
            "GET /api/sembol-kontrol/{sembol}": "Sembolün geçerliliğini kontrol eder.",
            "/docs": "API dokümantasyonu."
        }
    }
@app.post("/api/teknik-analiz", response_model=TechnicalAnalysisResponse)
async def teknik_analiz(
    request: Request,
    data: TechnicalAnalysisRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Belirtilen hisse senedi için teknik analiz yapar ve sonuçları döndürür.
    Grafikler istendiyse, base64 yerine grafik URL'lerini döndürür.
    """
    sembol = data.symbol.upper()
    logger.info(f"İşleniyor: /api/teknik-analiz Sembol={sembol}")

    try:
        baslangic_tarih = data.start_date
        bitis_tarih = data.end_date

        analysis_start_time = time.time()
        # ana_teknik_analiz artık chart_ids döndürüyor
        veri, son_veriler, fib_seviyeler, indicators, chart_ids = ana_teknik_analiz(
            sembol,
            baslangic_tarih,
            bitis_tarih,
            data.include_charts
        )
        analysis_process_time = (time.time() - analysis_start_time) * 1000
        logger.info(f"'{sembol}' için ana_teknik_analiz çağrısı tamamlandı ({analysis_process_time:.2f}ms).")

        if veri is None or son_veriler is None or fib_seviyeler is None or indicators is None:
             raise HTTPException(status_code=404, detail=f"'{sembol}' için teknik analiz verisi oluşturulamadı.")

        period = f"{veri.index[0].strftime('%Y-%m-%d')} ile {veri.index[-1].strftime('%Y-%m-%d')} arası"
        logger.debug(f"'{sembol}' için yanıt periyodu: {period}")

        latest_data = []
        if son_veriler is not None and not son_veriler.empty:
            for idx, row in son_veriler.iterrows():
                data_entry = {
                    'date': idx.strftime('%Y-%m-%d'),
                    'open': float(row['Open']) if not pd.isna(row['Open']) else None,
                    'high': float(row['High']) if not pd.isna(row['High']) else None,
                    'low': float(row['Low']) if not pd.isna(row['Low']) else None,
                    'close': float(row['Close']) if not pd.isna(row['Close']) else None,
                }
                # Opsiyonel alanlar (NaN kontrolü ile)
                for col in ['Volume', 'RSI', 'SMA_20', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']:
                    if col in row and not pd.isna(row[col]):
                        data_entry[col.lower().replace('_', '')] = float(row[col])
                latest_data.append(data_entry)
            logger.debug(f"'{sembol}' için son {len(latest_data)} günlük veri yanıt için formatlandı.")
        else:
             logger.warning(f"'{sembol}' için 'son_veriler' boş veya None geldi, yanıtta 'latest' boş olacak.")

        # Yanıtı oluşturmaya başla
        response_data = {
            "symbol": sembol,
            "period": period,
            "data": {
                "latest": latest_data
            },
            "indicators": indicators,
            "fibonacci_levels": {k: float(v) for k, v in fib_seviyeler.items()} if fib_seviyeler else {},
            "charts": None # Başlangıçta None
        }

        # Grafikler istendiyse ve chart_ids varsa, URL'leri oluştur
        if data.include_charts and chart_ids:
            chart_urls = {}
            base_url = str(request.base_url) # İstek yapılan base URL'yi al (örn: http://127.0.0.1:8000/)
            for key, chart_id in chart_ids.items():
                # key isimlerini daha anlamlı yapalım (örn: price_chart_id -> price)
                url_key = key.replace('_chart_id', '')
                chart_urls[url_key] = f"{base_url}api/chart/{chart_id}"
            response_data["charts"] = chart_urls
            logger.debug(f"'{sembol}' için grafik URL'leri yanıta eklendi: {chart_urls}")
        elif data.include_charts and not chart_ids:
             logger.warning(f"'{sembol}' için grafikler istendi ancak oluşturulamadı veya saklanamadı.")

        logger.info(f"'{sembol}' için /api/teknik-analiz yanıtı başarıyla oluşturuldu.")
        return response_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"'{sembol}' için /api/teknik-analiz endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: Analiz sırasında beklenmedik bir sorun oluştu.")

@app.get("/api/sembol-kontrol/{sembol}")
async def sembol_kontrol(
    sembol: str,
    api_key: str = Depends(get_api_key)
):
    """
    Verilen sembolün geçerli olup olmadığını kontrol eder
    """
    sembol_upper = sembol.upper()
    logger.info(f"İşleniyor: /api/sembol-kontrol Sembol={sembol_upper}")
    try:
        bugun = datetime.now().strftime('%Y-%m-%d')
        bir_ay_once = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Veriyi çekmeyi dene (Bu fonksiyon önbelleğe alınmadı, her seferinde kontrol eder)
        veri = get_stock_data(sembol_upper, bir_ay_once, bugun)
        
        # Veri yoksa veya boşsa
        if veri is None or veri.empty:
            # get_stock_data zaten logladı
            return {"valid": False, "message": f"Sembol bulunamadı veya veri çekilemedi: {sembol_upper}"}
        
        # Temel bilgileri almayı dene (Bu kısım da ağ isteği yapabilir, yavaş olabilir)
        short_name = sembol_upper # Varsayılan
        try:
            logger.debug(f"'{sembol_upper}' için yfinance Ticker info alınıyor...")
            ticker_info = yf.Ticker(sembol_upper).info
            short_name = ticker_info.get('shortName', sembol_upper)
            logger.debug(f"'{sembol_upper}' için Ticker bilgisi alındı. Kısa İsim: {short_name}")
        except Exception as info_e:
            logger.warning(f"'{sembol_upper}' için yfinance Ticker info alınırken hata: {info_e}. Sembol ismi kullanılacak.")
            # Hata durumunda devam et, sadece isim alınamadı.

        last_price = None
        if not veri.empty and 'Close' in veri.columns and not pd.isna(veri['Close'].iloc[-1]):
             last_price = float(veri['Close'].iloc[-1])

        response = {
            "valid": True,
            "symbol": sembol_upper,
            "name": short_name,
            "last_price": last_price,
            "data_available": True # Veri çekebildiysek True
        }
        logger.info(f"Sembol kontrolü başarılı: {sembol_upper}")
        return response
        
    except Exception as e:
        logger.error(f"'{sembol_upper}' için /api/sembol-kontrol sırasında beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: Sembol kontrolü sırasında beklenmedik bir sorun oluştu.")

###############################################################################
# 4.5) Saklanan Grafikleri Getirme Endpoint'i
###############################################################################
@app.get("/api/chart/{chart_id}")
async def get_chart(chart_id: str):
    """Saklanan grafiği chart_id ile getirir ve PNG olarak döndürür."""
    logger.debug(f"Grafik isteği alındı: chart_id={chart_id}")
    if chart_id not in chart_storage:
        logger.warning(f"Grafik bulunamadı: chart_id={chart_id}")
        raise HTTPException(status_code=404, detail="Chart not found")

    chart_info = chart_storage[chart_id]

    # Süresi dolmuş mu kontrol et
    if datetime.now() > chart_info["expires_at"]:
        logger.warning(f"Grafik süresi dolmuş: chart_id={chart_id}. Siliniyor.")
        del chart_storage[chart_id] # Süresi dolanı sil
        raise HTTPException(status_code=410, detail="Chart expired and removed") # 410 Gone daha uygun

    # Base64'ten içeriğe dönüştür ve uygun medya tipi ile döndür
    try:
        image_bytes = base64.b64decode(chart_info["data"])
        media_type = chart_info.get("media_type", "image/png")
        logger.debug(f"Grafik başarıyla bulundu ve decode edildi: chart_id={chart_id}, media_type={media_type}")
        return Response(content=image_bytes, media_type=media_type)
    except Exception as e:
        logger.error(f"Grafik decode edilirken hata oluştu: chart_id={chart_id}, Hata: {e}", exc_info=True)
        # Hatalı veriyi temizleyebiliriz
        del chart_storage[chart_id]
        raise HTTPException(status_code=500, detail="Error decoding chart data")

###############################################################################
# 5) Uygulamayı Çalıştırma Bloğu
###############################################################################
if __name__ == "__main__":
    # Düz komut akışı: port açmadan grafikleri oluştur ve göster
    symbol_arg = sys.argv[1].upper() if len(sys.argv) > 1 else None
    if not symbol_arg:
        print("Kullanım: python grafik.py THYAO")
        sys.exit(0)

    baslangic = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    bitis = datetime.now().strftime('%Y-%m-%d')

    veri = get_stock_data(symbol_arg, baslangic, bitis)
    if veri is None or veri.empty:
        print(f"Veri alınamadı: {symbol_arg}")
        sys.exit(1)

    veri = hesapla_ma(veri, 20, "SMA")
    veri = hesapla_ma(veri, 50, "SMA")
    veri = hesapla_ma(veri, 20, "EMA")
    veri = hesapla_ma(veri, 50, "EMA")
    veri = hesapla_rsi(veri)
    veri = hesapla_bollinger_bantlari(veri)
    veri = hesapla_macd(veri)
    fib = hesapla_fibonacci_seviyeleri(veri)

    fig1 = ciz_fiyat_grafigi(veri, symbol_arg)
    fig2 = ciz_bollinger_fibonacci_grafigi(veri, symbol_arg, fib)
    fig3 = ciz_teknik_gostergeler_grafigi(veri, symbol_arg)

    # Dosya kaydetmeden görüntüle
    fig1.show()
    fig2.show()

    fig3.show()
