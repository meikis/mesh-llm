import { RouterProvider } from '@tanstack/react-router'
import { AppProviders } from '@/app/providers/AppProviders'
import { router } from '@/app/router/router'

// Production builds default to live mesh data; dev keeps harness for playground work.
const defaultDataMode = import.meta.env.DEV ? 'harness' : 'live'

export function App() {
  return (
    <AppProviders initialDataMode={defaultDataMode}>
      <RouterProvider router={router} />
    </AppProviders>
  )
}
