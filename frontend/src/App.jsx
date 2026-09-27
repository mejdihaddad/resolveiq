// frontend/src/App.jsx
import React, { useState } from 'react'

// Generate current wall-clock date/time specifically in America/New_York (YYYY-MM-DDTHH:mm)
const getNYCDefaultDateTime = () => {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })

  const parts = formatter.formatToParts(new Date())
  const getPart = (type) => parts.find((p) => p.type === type)?.value
  const year = getPart('year')
  const month = getPart('month')
  const day = getPart('day')
  let hour = getPart('hour')
  if (hour === '24') hour = '00'
  const minute = getPart('minute')

  return `${year}-${month}-${day}T${hour}:${minute}`
}

export function App() {
  const [formData, setFormData] = useState({
    intakeDateTime: getNYCDefaultDateTime(),
    complaintType: 'HEAT/HOT WATER',
    locationType: 'RESIDENTIAL BUILDING',
    descriptor: 'APARTMENT ONLY',
    descriptor2: 'NO HEAT',
    borough: 'BROOKLYN',
    openDataChannelType: 'ONLINE',
    agency: 'HPD',
    includeSimilarCases: false, // Explicit caller control switch
  })

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [triageResult, setTriageResult] = useState(null)

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
    // Clear stale outputs as soon as any form input or toggle is modified
    setTriageResult(null)
    setError(null)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      // 1. Derive temporal features directly from NYC wall-clock input string
      if (!formData.intakeDateTime || !formData.intakeDateTime.includes('T')) {
        throw new Error('Please enter a valid intake date and time.')
      }

      const [datePart, timePart] = formData.intakeDateTime.split('T')
      const [year, month, day] = datePart.split('-').map(Number)
      const [hour, minute] = timePart.split(':').map(Number)

      if (isNaN(year) || isNaN(month) || isNaN(day) || isNaN(hour) || isNaN(minute)) {
        throw new Error('Invalid date/time components.')
      }

      // Displayed wall-clock hour in America/New_York (0-23)
      const created_hour = hour

      // Day of week: calculate from UTC year/month/day to avoid local browser timezone shift
      // JS: Sunday=0, Monday=1, ..., Saturday=6 -> Python: Monday=0, ..., Sunday=6
      const jsDay = new Date(Date.UTC(year, month - 1, day)).getUTCDay()
      const created_day_of_week = (jsDay + 6) % 7
      const created_is_weekend = created_day_of_week >= 5 ? 1 : 0

      // Naive ISO timestamp sent to backend (which localizes to America/New_York)
      const intakeTimeISO = `${formData.intakeDateTime}:00`

      // 2. Build Unified Triage Workflow Payload (POST /api/triage)
      const triagePayload = {
        descriptor: formData.descriptor,
        descriptor_2: formData.descriptor2 || null,
        location_type: formData.locationType,
        open_data_channel_type: formData.openDataChannelType,
        agency: formData.agency,
        created_hour,
        created_day_of_week,
        created_is_weekend,
        complaint_type: formData.complaintType,
        borough: formData.borough || null,
        intake_time: formData.includeSimilarCases ? intakeTimeISO : null,
        limit: 5,
        include_similar_cases: formData.includeSimilarCases,
      }

      // 3. Execute single consolidated workflow request
      const res = await fetch('/api/triage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(triagePayload),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(
          `Triage workflow failed (HTTP ${res.status}): ${
            JSON.stringify(errData.detail || errData)
          }`
        )
      }

      const data = await res.json()
      setTriageResult(data)
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 font-sans antialiased text-slate-900 bg-slate-50 min-h-screen">
      <header className="mb-8 pb-4 border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-800 tracking-tight">
          ResolveIQ — Triage & Risk Workbench
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          NYC 311 Heat & Hot Water Intake Scoring and Deterministic Precedent Retrieval (LangGraph Workflow)
        </p>
      </header>

      {error && (
        <div className="p-4 mb-6 text-sm bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6 items-start">
        {/* Left Column: Intake Form */}
        <section className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-800 mb-4 flex items-center justify-between">
            Intake Features
          </h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="intakeDateTime" className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1.5">
                Intake Date & Time (NYC Local Time)
              </label>
              <input
                type="datetime-local"
                id="intakeDateTime"
                name="intakeDateTime"
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-md bg-white text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={formData.intakeDateTime}
                onChange={handleInputChange}
                required
              />
              <p className="text-xs text-slate-500 mt-1">
                Parsed as America/New_York; derives hour, day of week, and case eligibility.
              </p>
            </div>

            <div>
              <label htmlFor="descriptor" className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1.5">
                Primary Descriptor
              </label>
              <select
                id="descriptor"
                name="descriptor"
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-md bg-white text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={formData.descriptor}
                onChange={handleInputChange}
                required
              >
                <option value="APARTMENT ONLY">APARTMENT ONLY</option>
                <option value="ENTIRE BUILDING">ENTIRE BUILDING</option>
              </select>
            </div>

            <div>
              <label htmlFor="descriptor2" className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1.5">
                Secondary Descriptor (Specific Defect)
              </label>
              <select
                id="descriptor2"
                name="descriptor2"
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-md bg-white text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={formData.descriptor2}
                onChange={handleInputChange}
              >
                <option value="">(None / General)</option>
                <option value="NO HEAT">NO HEAT</option>
                <option value="NO HEAT AND NO HOT WATER">NO HEAT AND NO HOT WATER</option>
                <option value="NO HOT WATER">NO HOT WATER</option>
              </select>
            </div>

            <div>
              <label htmlFor="borough" className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1.5">
                Borough (Similar-Cases Filter Only)
              </label>
              <select
                id="borough"
                name="borough"
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-md bg-white text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={formData.borough}
                onChange={handleInputChange}
              >
                <option value="">All Boroughs (Unfiltered)</option>
                <option value="BRONX">BRONX</option>
                <option value="BROOKLYN">BROOKLYN</option>
                <option value="MANHATTAN">MANHATTAN</option>
                <option value="QUEENS">QUEENS</option>
                <option value="STATEN ISLAND">STATEN ISLAND</option>
              </select>
              <p className="text-xs text-slate-500 mt-1">
                Strictly excluded from Model 1 risk scoring; used solely to filter historical case search.
              </p>
            </div>

            <div>
              <label htmlFor="openDataChannelType" className="block text-xs font-semibold uppercase tracking-wider text-slate-600 mb-1.5">
                Submission Channel
              </label>
              <select
                id="openDataChannelType"
                name="openDataChannelType"
                className="w-full px-3 py-2 text-sm border border-slate-300 rounded-md bg-white text-slate-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={formData.openDataChannelType}
                onChange={handleInputChange}
              >
                <option value="ONLINE">ONLINE</option>
                <option value="PHONE">PHONE</option>
                <option value="MOBILE">MOBILE</option>
                <option value="OTHER">OTHER</option>
              </select>
            </div>

            {/* Explicit Control Switch for Precedent Retrieval */}
            <div className="flex items-start gap-3 p-3 bg-slate-50 border border-slate-200 rounded-md">
              <input
                type="checkbox"
                id="includeSimilarCases"
                name="includeSimilarCases"
                checked={formData.includeSimilarCases}
                onChange={handleInputChange}
                className="mt-1 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
              />
              <label htmlFor="includeSimilarCases" className="text-xs text-slate-700 leading-relaxed cursor-pointer select-none">
                <span className="font-semibold block text-slate-800">Retrieve Similar Precedent Cases</span>
                Query Qdrant collection-01 for historical precedents. When unchecked, vector retrieval is bypassed.
              </label>
            </div>

            <button
              type="submit"
              className="w-full py-2.5 px-4 text-sm font-semibold rounded-md bg-blue-600 hover:bg-blue-700 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
              disabled={loading}
            >
              {loading ? 'Executing Workflow...' : 'Execute Triage Workflow'}
            </button>
          </form>
        </section>

        {/* Right Column: Triage Results */}
        <div className="space-y-6">
          {triageResult && (
            <>
              {/* Risk Scoring & Workflow Decision Panel */}
              <section className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                  <span className="text-lg font-semibold text-slate-800">
                    Model 1 Intake Risk Score
                  </span>
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-block px-2.5 py-1 rounded-full text-xs font-semibold tracking-wider ${
                        triageResult.routing_decision === 'scored_and_retrieved_precedents'
                          ? 'bg-blue-100 text-blue-800 border border-blue-300'
                          : 'bg-slate-100 text-slate-700 border border-slate-300'
                      }`}
                    >
                      {triageResult.routing_decision === 'scored_and_retrieved_precedents'
                        ? 'Routing: Scored & Retrieved'
                        : triageResult.routing_decision === 'scored_retrieval_unavailable'
                          ? 'Routing: Retrieval Unavailable'
                          : triageResult.routing_decision === 'scored_retrieval_failed'
                            ? 'Routing: Retrieval Failed'
                            : 'Routing: Scored Only (Bypassed)'}
                    </span>
                    <span
                      className={`inline-block px-2.5 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${
                        triageResult.above_frozen_cutoff
                          ? 'bg-red-100 text-red-800 border border-red-300'
                          : 'bg-green-100 text-green-800 border border-green-300'
                      }`}
                    >
                      {triageResult.above_frozen_cutoff
                        ? 'Above Cutoff (>= 0.28)'
                        : 'Below Cutoff (< 0.28)'}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-6 p-4 bg-slate-50 border border-slate-200 rounded-md mb-4">
                  <div>
                    <div className="text-4xl font-extrabold text-slate-900 font-mono">
                      {triageResult.risk_score.toFixed(4)}
                    </div>
                    <div className="text-xs font-medium text-slate-500 mt-1">
                      Uncalibrated ranking score
                    </div>
                  </div>
                  <div className="text-sm text-slate-600 space-y-0.5 border-l border-slate-200 pl-6">
                    <div><strong>Operating Cutoff:</strong> {triageResult.operating_cutoff}</div>
                    <div><strong>Workflow Path:</strong> {triageResult.routing_decision}</div>
                  </div>
                </div>

                <div className="bg-amber-50 border-l-4 border-amber-500 p-3.5 text-xs text-amber-900 rounded-r-md">
                  <strong>Model Notice:</strong> {triageResult.disclaimer}
                </div>
              </section>

              {(triageResult.retrieval_status === 'unavailable' || triageResult.retrieval_status === 'failed') && (
                <section className="bg-amber-50 border border-amber-200 rounded-lg p-4 shadow-sm">
                  <p className="text-sm font-semibold text-amber-900">
                    {triageResult.retrieval_status === 'unavailable'
                      ? 'Similar-case retrieval temporarily unavailable'
                      : 'Similar-case retrieval could not be completed'}
                  </p>
                  <p className="text-sm text-amber-800 mt-1">
                    {triageResult.retrieval_message} The risk score above is still available.
                  </p>
                </section>
              )}

              {/* Similar Cases Panel (Rendered only when retrieval was requested) */}
              {triageResult.similar_cases !== null ? (
                <section className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-lg font-semibold text-slate-800">
                      Similar Historical Precedents
                    </span>
                    <span className="text-xs text-slate-500 bg-slate-100 px-2.5 py-1 rounded">
                      Collection-01 {formData.borough ? `(${formData.borough})` : ''}
                    </span>
                  </div>

                  <p className="text-xs text-slate-600 mb-4">
                    {triageResult.similar_cases.length} similar complaints retrieved under temporal admissibility (created before intake).
                  </p>

                  {triageResult.similar_cases.length === 0 ? (
                    <p className="text-sm text-slate-500 italic py-4">
                      No historical cases created prior to this intake timestamp were found.
                    </p>
                  ) : (
                    <div className="space-y-4">
                      {triageResult.similar_cases.map((c) => (
                        <div key={c.ticket} className="border border-slate-200 rounded-md p-4 bg-white hover:border-slate-300 transition-colors">
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-bold text-slate-800 text-sm">
                              Ticket #{c.ticket}
                            </span>
                            <span className="text-xs font-semibold bg-slate-100 text-slate-700 px-2 py-1 rounded border border-slate-200">
                              Cosine similarity: {c.similarity_score.toFixed(4)}
                            </span>
                          </div>

                          <div className="text-xs text-slate-600 grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 my-2">
                            <div><strong>Borough:</strong> {c.borough}</div>
                            <div><strong>Shape:</strong> {c.shape_slug}</div>
                            <div><strong>Created:</strong> {c.created_date}</div>
                            <div><strong>Descriptor:</strong> {c.descriptor} {c.descriptor_2 ? `• ${c.descriptor_2}` : ''}</div>
                          </div>

                          <div className="border-t border-dashed border-slate-200 pt-3 mt-2">
                            {c.is_historical_evidence ? (
                              <div className="text-xs text-slate-800 space-y-1">
                                <div><strong>Outcome:</strong> {c.status} ({c.closed_date})</div>
                                {c.resolution_description && (
                                  <div className="text-xs text-slate-600 bg-slate-50 p-2 rounded border border-slate-100">
                                    {c.resolution_description}
                                  </div>
                                )}
                              </div>
                            ) : (
                              <div>
                                <span className="inline-block text-xs bg-slate-100 text-slate-500 px-2.5 py-1 rounded font-medium border border-slate-200">
                                  Outcome Masked (Not resolved prior to intake time)
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              ) : (
                <section className="bg-slate-50 border border-slate-200 rounded-lg p-5 text-center text-xs text-slate-500">
                  Precedent retrieval was skipped (checkbox was disabled). Vector database and embedding calls were bypassed.
                </section>
              )}
            </>
          )}

          {!triageResult && !loading && (
            <section className="bg-white border border-slate-200 rounded-lg p-12 text-center shadow-sm">
              <h3 className="text-base font-medium text-slate-700 mb-1">
                Awaiting Intake Submission
              </h3>
              <p className="text-sm text-slate-500 max-w-md mx-auto">
                Fill out the complaint features on the left and click "Execute Triage Workflow"
                to run Model 1 risk scoring and optional precedent case retrieval.
              </p>
            </section>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
