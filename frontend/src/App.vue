<script setup>
import { computed, onMounted, ref } from 'vue'

const summary = ref(null)
const tournament = ref([])
const fixtures = ref([])
const isLoading = ref(true)
const errorMessage = ref('')

const percent = (value) => {
  if (value === null || value === undefined) return '-'
  const normalized = value <= 1 ? value * 100 : value
  return `${normalized.toFixed(normalized < 10 ? 1 : 0)}%`
}

const formatDate = (date) =>
  new Intl.DateTimeFormat('ko-KR', {
    month: 'short',
    day: 'numeric',
    weekday: 'short',
  }).format(new Date(`${date}T00:00:00`))

const formatNumber = (value) => new Intl.NumberFormat('ko-KR').format(value ?? 0)

const displayTeam = (team) => {
  if (['South Korea', 'Korea Republic'].includes(team)) return '대한민국'
  return team
}

const topTeams = computed(() =>
  [...tournament.value].sort((a, b) => b.winnerProb - a.winnerProb).slice(0, 10),
)

const korea = computed(
  () =>
    tournament.value.find((team) =>
      ['South Korea', 'Korea Republic'].includes(team.team),
    ) ??
    summary.value?.southKorea ??
    null,
)

const maxWinnerProbability = computed(() => topTeams.value[0]?.winnerProb ?? 0)

const resultLabel = (result) =>
  ({
    HOME_WIN: '홈 승',
    DRAW: '무승부',
    AWAY_WIN: '원정 승',
  })[result] ?? result

const resultClass = (result) =>
  ({
    HOME_WIN: 'result-home',
    DRAW: 'result-draw',
    AWAY_WIN: 'result-away',
  })[result] ?? ''

onMounted(async () => {
  try {
    const responses = await Promise.all([
      fetch('/data/dashboard_summary.json'),
      fetch('/data/tournament_simulation.json'),
      fetch('/data/fixture_predictions.json'),
    ])

    if (responses.some((response) => !response.ok)) {
      throw new Error('대시보드 데이터를 불러오지 못했습니다.')
    }

    ;[summary.value, tournament.value, fixtures.value] = await Promise.all(
      responses.map((response) => response.json()),
    )
  } catch (error) {
    errorMessage.value = error.message
  } finally {
    isLoading.value = false
  }
})
</script>

<template>
  <main class="dashboard-shell">
    <div class="dashboard">
      <header class="hero">
        <div>
          <div class="eyebrow">
            <span class="live-dot"></span>
            10,000회 시뮬레이션 기반
          </div>
          <h1>2026 World Cup <span>Predictor</span></h1>
          <p>
            국가대표 경기 데이터를 기반으로 승부와 토너먼트 진출 가능성을 예측합니다.
          </p>
        </div>
        <div class="hero-mark" aria-hidden="true">
          <span>26</span>
          <small>WORLD CUP</small>
        </div>
      </header>

      <div v-if="isLoading" class="state-panel">
        <div class="spinner"></div>
        <p>예측 데이터를 불러오는 중입니다.</p>
      </div>

      <div v-else-if="errorMessage" class="state-panel error-panel">
        <strong>데이터 로딩 오류</strong>
        <p>{{ errorMessage }}</p>
      </div>

      <template v-else>
        <section class="overview-grid" aria-label="모델 및 대한민국 예측 요약">
          <article class="panel model-card">
            <div class="panel-heading">
              <div>
                <p class="section-kicker">MODEL OVERVIEW</p>
                <h2>모델 요약</h2>
              </div>
              <span class="status-badge">ACTIVE</span>
            </div>

            <div class="model-name">
              <span class="model-icon">ML</span>
              <div>
                <strong>{{ summary.modelSummary.modelName }}</strong>
                <span>Multiclass Classification</span>
              </div>
            </div>

            <div class="metric-grid">
              <div class="metric">
                <span>정확도</span>
                <strong>{{ percent(summary.modelSummary.accuracy) }}</strong>
              </div>
              <div class="metric">
                <span>Log Loss</span>
                <strong>{{ summary.modelSummary.logLoss }}</strong>
              </div>
              <div class="metric">
                <span>학습 데이터</span>
                <strong>{{ formatNumber(summary.modelSummary.trainSize) }}</strong>
              </div>
              <div class="metric">
                <span>테스트 데이터</span>
                <strong>{{ formatNumber(summary.modelSummary.testSize) }}</strong>
              </div>
            </div>

            <p class="model-note">{{ summary.modelSummary.note }}</p>
          </article>

          <article class="panel korea-card">
            <div class="korea-glow"></div>
            <div class="panel-heading">
              <div>
                <p class="section-kicker">KOREA REPUBLIC · GROUP {{ korea?.group }}</p>
                <h2>대한민국 진출 확률</h2>
              </div>
              <span class="flag" aria-label="대한민국">KR</span>
            </div>

            <div class="korea-primary">
              <div>
                <span>32강 진출</span>
                <strong>{{ percent(korea?.roundOf32Prob) }}</strong>
              </div>
              <div
                class="probability-ring"
                :style="{ '--probability': `${(korea?.roundOf32Prob ?? 0) * 360}deg` }"
              >
                <span>R32</span>
              </div>
            </div>

            <div class="korea-stages">
              <div>
                <span>16강</span>
                <strong>{{ percent(korea?.roundOf16Prob) }}</strong>
              </div>
              <div>
                <span>8강</span>
                <strong>{{ percent(korea?.quarterFinalProb) }}</strong>
              </div>
              <div>
                <span>4강</span>
                <strong>{{ percent(korea?.semiFinalProb) }}</strong>
              </div>
              <div>
                <span>우승</span>
                <strong>{{ percent(korea?.winnerProb) }}</strong>
              </div>
            </div>
          </article>
        </section>

        <section class="panel section-panel">
          <div class="section-header">
            <div>
              <p class="section-kicker">FIXTURE FORECAST</p>
              <h2>예정 경기 승부 예측</h2>
            </div>
            <span class="count-badge">{{ fixtures.length }} MATCHES</span>
          </div>

          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>날짜</th>
                  <th>홈팀</th>
                  <th>원정팀</th>
                  <th>홈승 확률</th>
                  <th>무승부 확률</th>
                  <th>원정승 확률</th>
                  <th>예측 결과</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(fixture, index) in fixtures"
                  :key="`${fixture.date}-${fixture.homeTeam}-${fixture.awayTeam}`"
                  :class="{ 'korea-row': [fixture.normalizedHomeTeam, fixture.normalizedAwayTeam].includes('Korea Republic') }"
                >
                  <td>
                    <span class="match-number">{{ String(index + 1).padStart(2, '0') }}</span>
                    {{ formatDate(fixture.date) }}
                  </td>
                  <td class="team-name">{{ displayTeam(fixture.homeTeam) }}</td>
                  <td class="team-name">{{ displayTeam(fixture.awayTeam) }}</td>
                  <td>{{ percent(fixture.homeWinProb) }}</td>
                  <td>{{ percent(fixture.drawProb) }}</td>
                  <td>{{ percent(fixture.awayWinProb) }}</td>
                  <td>
                    <span class="result-badge" :class="resultClass(fixture.predictedResult)">
                      {{ resultLabel(fixture.predictedResult) }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="panel section-panel">
          <div class="section-header">
            <div>
              <p class="section-kicker">CHAMPIONSHIP ODDS</p>
              <h2>우승 확률 Top 10</h2>
            </div>
            <span class="count-badge">SIMULATION RANKING</span>
          </div>

          <div class="ranking-list">
            <article
              v-for="(team, index) in topTeams"
              :key="team.team"
              class="ranking-row"
              :class="{ podium: index < 3 }"
            >
              <span class="rank">{{ String(index + 1).padStart(2, '0') }}</span>
              <div class="team-symbol">{{ team.team.slice(0, 2).toUpperCase() }}</div>
              <div class="ranking-team">
                <strong>{{ displayTeam(team.displayTeam || team.team) }}</strong>
                <span>GROUP {{ team.group }}</span>
              </div>
              <div class="ranking-bar">
                <span
                  :style="{
                    width: `${(team.winnerProb / maxWinnerProbability) * 100}%`,
                  }"
                ></span>
              </div>
              <div class="stage-chance">
                <span>결승</span>
                <strong>{{ percent(team.finalProb) }}</strong>
              </div>
              <div class="winner-chance">
                <span>우승</span>
                <strong>{{ percent(team.winnerProb) }}</strong>
              </div>
            </article>
          </div>
        </section>

        <footer>
          <span>2026 WORLD CUP PREDICTOR</span>
          <p>예측 결과는 통계 모델에 기반한 시뮬레이션이며 실제 결과와 다를 수 있습니다.</p>
        </footer>
      </template>
    </div>
  </main>
</template>
