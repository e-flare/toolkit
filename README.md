<p align="right">English | <a href="./README_CN.md">简体中文</a></p>  


<p align="center">
  <img src=".docs/figures/camera-lens.gif" width="12.5%" align="center">

  <h1 align="center">
    <strong>Learning to Remove Lens Flare in Event Camera
  </h1>

  <p align="center">
    <a href="https://openreview.net/profile?id=~Haiqian_Han1" target="_blank"><strong>Haiqian Han</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ldkong.com/" target="_blank"><strong>Lingdong Kong</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://jianing-li.github.io/" target="_blank"><strong>Jianing Li</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://alanliangc.github.io/" target="_blank"><strong>Ao Liang</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://openreview.net/profile?id=~Chengtao_Zhu1" target="_blank"><strong>Chengtao Zhu</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://openreview.net/profile?id=~Jiacheng_Lyu1" target="_blank"><strong>Jiacheng Lyu</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ipal.cnrs.fr/lai-xing-ng/" target="_blank"><strong>Lai Xing Ng</strong></a><br>   
    <a href="https://www.au.tsinghua.edu.cn/info/1080/3178.htm" target="_blank"><strong>Xiangyang Ji</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.comp.nus.edu.sg/cs/people/ooiwt/" target="_blank"><strong>Wei Tsang Ooi</strong></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://ipal.cnrs.fr/benoit-cottereau-personal-page/" target="_blank"><strong>Benoit R. Cottereau</strong></a>    
  </p>

  <p align="center">
    <a href="" target='_blank'>
      <img src="https://img.shields.io/badge/Paper-%F0%9F%93%96-darkred">
    </a>&nbsp;
    <a href="http://e-flare.github.io/" target='_blank'>
      <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-blue">
    </a>&nbsp;
    <a href="https://huggingface.co/datasets/e-flare/data" target='_blank'>
      <img src="https://img.shields.io/badge/Dataset-%F0%9F%94%97-green">
    </a>&nbsp;
    <a href="" target='_blank'>
      <img src="https://visitor-badge.laobi.icu/badge?page_id=e-flare.toolkit">
    </a>
  </p>


| <img src=".docs/figures/teaser.png" alt="Teaser" width="100%"> |
| :-: |

Event cameras have the potential to revolutionize vision systems with their high temporal resolution and dynamic range, yet they remain susceptible to lens flare, a fundamental optical artifact that causes severe degradation. In event streams, this optical artifact forms a complex, spatio-temporal distortion that has been largely overlooked.

We present the **first** systematic framework for removing lens flare from event camera data. 
- We first establish the theoretical foundation by deriving a physics-grounded forward model of the non-linear suppression mechanism.
- This insight enables the creation of the E-Deflare Benchmark, a comprehensive resource featuring a large-scale simulated training set, E-Flare-2.7K, and the first-ever paired real-world test set, E-Flare-R, captured by our novel optical system.
- Empowered by this benchmark, we design E-DeflareNet, which achieves state-of-the-art restoration performance.

Extensive experiments validate our approach and demonstrate clear benefits for downstream tasks.

### :books: Citation
If you find this work helpful for your research, please kindly consider citing our papers:

```bibtex
@article{han2025e-deflare,
    title   = {Learning to Remove Lens Flare in Event Camera},
    author  = {Haiqian Han and Lingdong Kong and Jianing Li and Ao Liang and Chengtao Zhu and Jiacheng Lyu and Lai Xing Ng and Xiangyang Ji and Wei Tsang Ooi and Benoit R. Cottereau},
    journal = {arXiv preprint arXiv:2512.xxxxx}
    year    = {2025}
}
```


