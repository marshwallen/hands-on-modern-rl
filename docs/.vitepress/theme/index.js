import DefaultTheme from 'vitepress/theme'
import './style.css'
import Layout from './Layout.vue'
import NavCard from './components/NavCard.vue'
import NavGrid from './components/NavGrid.vue'
import StepBar from './components/StepBar.vue'
import Mermaid from 'vitepress-plugin-mermaid/Mermaid.vue'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp(ctx) {
    DefaultTheme.enhanceApp?.(ctx)
    ctx.app.component('NavCard', NavCard)
    ctx.app.component('NavGrid', NavGrid)
    ctx.app.component('StepBar', StepBar)
    ctx.app.component('Mermaid', Mermaid)
  }
}
