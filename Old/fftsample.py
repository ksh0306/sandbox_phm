import numpy as np
import matplotlib.pyplot as plt

def log_base_16_transform(data):
    """
    밑수가 2^4(16)인 로그 변환 모듈.
    0 입력 시 발산을 막기 위해 최소값(epsilon)을 더함.
    """
    epsilon = 1e-9
    # log16(x) = log2(x) / log2(16) = log2(x) / 4
    return np.log2(data + epsilon) / 4

# 1. 가상 데이터 생성 (전류 파형 모사: 60Hz 기본파 + 180Hz 고조파 + 노이즈)
fs = 1000  # 샘플링 주파수 1000Hz
t = np.linspace(0, 1, fs, endpoint=False)
# 0 ~ 65535 범위의 16-bit ADC 데이터 모사
raw_signal = (30000 * np.sin(2 * np.pi * 60 * t) + 
              5000 * np.sin(2 * np.pi * 180 * t) + 32768)
raw_signal = np.clip(raw_signal, 0, 65535).astype(np.int32)

# 2. 로그 변환 (밑수 16)
log_scaled_data = log_base_16_transform(raw_signal)

# 3. FFT 수행
n = len(log_scaled_data)
fft_result = np.fft.fft(log_scaled_data)
fft_freq = np.fft.fftfreq(n, 1/fs)

# 절반만 추출 (대칭성) 및 크기(Magnitude) 계산
magnitude = np.abs(fft_result)[:n//2]
frequencies = fft_freq[:n//2]

# 4. 시각화
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t[:200], raw_signal[:200], label='Raw ADC Data')
plt.title("Time Domain: Raw Signal (0-65535)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude, color='red')
plt.title("Frequency Domain: After Log16 Transform & FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()