# Overview

Are you ready to embark on a journey that pushes the boundaries of astronomical data analysis?

The Ariel Data Challenge 2024 invites you to develop machine learning models to solve one of the most formidable challenges in the field—extracting faint exoplanetary signals from simulated observations of the upcoming ESAe Ariel Mission!

태양계 밖에 있는 행성의 자극적 관측으로 얻은 희미한 시그널 추출하기

---

# Description

The discovery of exoplanets—planets orbiting stars other than our Sun—has transformed our cosmic perspective, challenging conventional notions about Earth’s uniqueness and the potential for life elsewhere. As of today, we are aware of over 5,600 exoplanets. Detecting these worlds is the initial step; we must also comprehend and characterise their nature by studying their atmospheres. In 2029, [ESA Ariel Mission](https://arielmission.space/index.php/data-challenges/) will conduct the first comprehensive study of 1,000 extrasolar planets in our galactic neighbourhood.

태양계 밖에 있는 행성 중에 지구처럼 사람이 살 수 있을만한 행성을 찾기 위한 거대 프로젝트가 진행중인 것 같음. → ESA Ariel Mission

현재 초기 단계인데, 지금까지 파악된 태양계 밖에 있는 행성은 5,600개 이상으로 각 행성의 대기를 연구함으로써 이 행성들의 자연적인 특성을 파악해야 함.

Observing these atmospheres is one of the hardest data-analysis problems in contemporary astronomy. When an exoplanet transits its host start in our line of sight, a tiny fraction of starlight (50-200 photons per million) passes through the planet’s atmospheric annulus and interacts with its chemistry, clouds, and winds. These faint signals typically range from 50ppm (for Super-Earth like planets) to 200ppm (for Jupiter like planets) in magnitude and are regularly corrupted by the noise of the instrument. A major component of this noise is due to the inevitable vibration of the spacecraft in space, known as ‘jitter noise’. This noise arises from the difficulties of maintaining precise pointing in low-gravity environments, as the spacecraft relies on spinning momentum wheels for stability. Akin to taking long-exposure images with a shaky hand, this noise poses a far greater challenge than the motion blur encountered in commercial photography applications. The photometric variation (~200ppm) caused by jitter noise alone is comparable to the variation exhibited by the planetary signal we aim to detect, undermining signals from small planets like Earths and super-Earths. Coupled with other sources of correlated and uncorrelated noisese, it is proving difficult for us to achieve the strict technical requirement of the Ariel Payload design.

행성의 대기를 관측하는 것은 현대 천문학에서 가장 어려운 데이터 분석 중 하나

행성이 우리 시야에서의 호스트 별을 지날 때 희미한 빛이 대기 고리를 지나 화학적, 구름, 바람의 상호작용이 나타남.

이때의 시그널이 보통 50ppm~200ppm인데 기계 노이즈에 의해 오염됨: 주된 오염은 우주선에 의한 어쩔 수 없는 진동 (보통 jitter nosie 라고 함) → 이 노이즈는 중력이 적은 환경에서 정확하게 포인팅을 유지하는 걸 어렵게 함

이러한 노이즈들이 결합되어 있어서 지금 프로젝트 진행이 어려움

The task of this competition is to extract the atmospheric spectra from each observation, with an estimate of its level of uncertainly. In order to obtain such a spectrum, we require the participant to detrend a large number of sequential 2D images of the spectral focal plane taken over several hours of observing the exoplanet as it eclipses its host star.

Performing this detrending process to extract atmospheric spectra and their associated errorbars from raw observational data is a crucial and common prerequisite step for any modern astronomical instrument before the data can undergo scientific analysis.

할 일은 불확실한 레벨 측정으로 얻어진 각 관측에서 대기 스펙트럼을 추출하는 것

행성이 호스트 별을 가리는 순간 몇 시간 넘게 관측되는 스펙트럼 중심의 면에서 얻어지는 많은 수의 2차원의 순차적인 이미지에서 추세 제거를 했으면 좋겠음

대기 스펙트럼에서 추출하여 디트렌딩하는 작업과 실제 관측 데이터에 대한 오차는 계속 생각하면서 작업

최근의 천문학 기기의 데이터를 사용하기 전의 전제조건은 과학적 분석을 거쳤다는 것

---

# Possible Approaches

![inbox_18942071_d43f0c5cd9f93fc2c334ec42f0cdd95b_data_reduction.jpg](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa1832e2-002b-401d-88c9-e68af8ba7241/4081662a-4a2f-4216-b1a6-8214dff327af/inbox_18942071_d43f0c5cd9f93fc2c334ec42f0cdd95b_data_reduction.jpg)

This is a multimodal supervised learning task. Participants can choose to detrend this jitter noise in either modality (i.e. the image, time or spectral domains) or combinations thereof. Each modality bears different advantages. Here we outline two common training strategies.

다방면의 지도 학습 방법이 있음 → 이미지, 시간, 스펙트럼 도메인 중 하나 선택하여 디드렌링하거나 다 사용하거나

- Approach 1: Train directly on the full 3D data cube and extract the corresponding sepctra. This approach leverages the rich information content but as a consequence requires a lot of computing resources (See Image → Spectral Domain on the above figure).

방법1: 전체 3D 데이터 큐브를 바로 학습시키고 일치하는 스펙트럼 추출.

많은 정보에 대한 영향을 미칠 수 있지만 컴퓨팅 자원이 너무 많이 필요함.

- Approach 2: Make the data lighter by summing up the fluxes along the pixel y-axis, for each wavelength, resulting in 2D images of dimension (N_times, N_wavelengths), and transform the images in order to enhance transit depth variations between wavelengths.

방법2: 데이터를 가볍게 만들기 (각 파동 길이마다 픽셀 y축인 fluxes의 합) → 2차원 이미지 생성(N_시간, N_파동길이) 및 변환

파동 길이 간의 깊이 변화 전환을 향상시키기 위함

However, neither approach is optimal for denoising jitter time series and we anticipate the winning solutions to include information from all three domains.

그렇지만, 둘 중에 어느것도 jitter 시계열의 노이즈 제거에 최적화된 것은 아님.

세 가지의 모든 도메인의 정보를 포함하는 해결방법을 찾았으면 좋겠음

---

# Evaluation

This competition evaluates predicted spectra ($\mu$_user) and corresponding uncertainties ($\sigma$_user) for different wavelengths against the ground truth pixel level spectrum (y) using the Gaussian Log-likelihood (GLL) function.

$$
GLL = -\frac{1}{2}(log(2\pi) + log(\sigma^2_{user}) + \frac{(y-\mu_{user})^2}{\sigma^2_{user}}
$$

The GLL values from each pair will be summed across all wavelengths and across the entire test set to produce a final GLL value (L). The final GLL value will be transformed into a score using the following conversion function:

$$
score = \frac{L - L_{ref}}{L_{ideal} - L_{ref}}
$$

We define L_ideal as the case where the submission perfectly matches the ground truth values, with an uncertainty of 10 parts per million (ppm). This ideal case is defined based on Ariel’s Stability Requirement. For L_ref is defined using the mean and variance of the training dataset as its prediction for all instances.

The score will return a float in the interval [0,1], with higher scores corresponding to better performing models. Any score below 0 will be treated as 0.

The full metric implementation is available here. (link)

---

# Submission File

You must predict a mean and uncertainty for each `planet_id` . An example submission file is included in the Data Files. Each submission row must include 567 columns, so we will not attempt to provide an example here. The leftmost column must be the `planet_id` , the next 283 columns must be the spectra, and the remaining columns the uncertainties.

---

# Timeline

- August 1, 2024 - Start Date.
- October 24, 2024 - Entry Deadline. You must accept the competition rules before this date in order to compete.
- October 24, 2024 - Team Merger Deadline. This is the last day participants may join or merge teams.
- October 31, 2024 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

---

# Dataset Description

태양계외 행성의 화학적 특성화는 천문학적인 관점에서 가장 활발한 프로젝트 중 하나. ARIEL 미션은 대략 1,000개의 행성이 호스트 별의 앞을 지날 때의 관측 데이터를 모으는 것. 좋은 기기를 가지고 있어도 결과 데이터는 한정적인 광자의 수와 많은 노이즈를 가질 것. 이런 시뮬레이션 데이터에서 행성 대기의 화학적 스펙트럼을 추출하는 것이 챌린지!

실제 제공되는 데이터로 테스트를 진행하고 나서 나중에 평가할 때 800개의 다른 행성 데이터로 평가할 것

여러 개의 태양계외 행성 시뮬레이션은 실제 행성을 기반으로 이루어짐. 이런 경우는 채점되지 않음.

## Metadata Files

- [train/test]_adc_info.csv
    - 데이터의 원래 dynamic 범위를 복원하기(for restoring) 위한 아날로그-디지털 변환(ADC) 파라미터 (gain and offer),  행성 시뮬레이션에 어떤 별을 사용했는지 식별하기 위한 `star` 칼럼 포함
- train_labels.csv
    - 실제 스펙트럼 (ground truth)
- axis_info.csv
    - 두 가지의 기기의 축 정보 (Axis information for both instruments
- wavelength.csv
    - 데이터셋의 실제 스펙트럼의 파형 길이 그리드 (The wavelength grid for each ground truth spectrum in the dataset)

## Signal Files

데이터셋 구성: 시계열 이미지 두 개의 개별 기기 + 교정 데이터

Ariel 데이터는 여러 개의 관측 기기 포함, 각 기기는 스펙트럼 대역폭와 관측 모드가 다름.

FGS1는 Ariel’s Fine Gudianece System(FGS)의 첫 번째 채널. FGS의 주요 태스크는 위성 센터링, 포커싱,  가이드. 가시 스펙트럼에서 타겟 별의 높은 정확도의 광도 측정도 제공. 0.06~0.08 $\mu m$의 민감도.

AIRS-CH0는 Ariel InfraRed Spectrometer(AIRS)의 첫 번째 채널(CH0).
