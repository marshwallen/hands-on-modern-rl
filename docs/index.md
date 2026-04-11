<script setup>
import { useRouter } from 'vitepress'
import { onMounted } from 'vue'

const router = useRouter()

onMounted(() => {
  // VitePress base needs to be considered
  router.go('/hands-on-modern-rl/preface/intro')
})
</script>
