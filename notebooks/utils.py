import string
from pandas import read_sql


def create_queries(customer, dsn, train_start='2017-01-01', train_end='2018-01-01', response_end='2019-01-01'):
    # first make the drug queries
    count_drugs = read_sql("""
    select gen_drug_name, count(distinct member_id) ct_members
    from {customer}.rx_claim rx
    where is_bh or is_chronic
    group by 1
    order by 2 desc
    """.format(customer=customer), dsn)

    rx_maker = """
    ,count(distinct case when gen_drug_name = '{drug}' then fill_dt end) as {drug_name}_fills
    """

    # only the 400 most popular drugs for each customer
    drugs = [
        x for x in count_drugs["gen_drug_name"].values[:400].tolist() if x is not None and "|" not in x
    ]

    def replace(s):
        for x in string.punctuation + " ":
            s = s.replace(x, "")
        return s.lower()

    drug_names = list(map(replace, drugs))
    drugs_strings = "".join([rx_maker.format(drug=drugs[i], drug_name=drug_names[i]) for i in range(len(drugs))])
    
    # now we have drug names, make the indiv feature queries
    med_query = """
    ,
    medical as (
        select members.member_id,
      min(case when bh_conds_cnt_all > 0 then datediff(day, claim_from_dt, '2019-05-01') else 365 end) as days_since_mh_dx,
      min(case when svc_cat in ('01_IP_Acute', '04_OP_Visit_ER')
      then datediff(day, claim_from_dt, '2019-05-01') else 365 end) as days_since_er_ip,
      count(distinct case when bh_conds_cnt_all > 0 then claim_from_dt end) as bh_visits,
      count(distinct case when svc_cat in ('04_OP_Visit_ER', '01_IP_Acute') then claim_from_dt end) as ip_er_visit_count,
      count(distinct case when svc_cat in ('04_OP_Visit_ER', '01_IP_Acute') and bh_conds_cnt > 0 then claim_from_dt end) as ip_er_bh_visit_count,
      count(distinct case when bh_conditions ilike '%bipolar%' then claim_from_dt end) as mh_bipolar_dx,
      count(distinct case when bh_conditions ilike '%schizophrenia%' then claim_from_dt end) as mh_schizo_dx,
      count(distinct case when bh_conditions ilike '%adhd%' then claim_from_dt end) as mh_adhd_dx,
      count(distinct case when bh_conditions ilike '%anxiety%' then claim_from_dt end) as mh_anxiety_dx,
      count(distinct case when bh_conditions ilike '%conduct%' then claim_from_dt end) as mh_conduct_dx,
      count(distinct case when bh_conditions ilike '%depression%' then claim_from_dt end) as mh_depression_dx,
      count(distinct case when bh_conditions ilike '%eating%' then claim_from_dt end) as mh_eating_dx,
      count(distinct case when bh_conditions ilike '%ocd%' then claim_from_dt end) as mh_ocd_dx,
      count(distinct case when bh_conditions ilike '%otherbh%' then claim_from_dt end) as mh_otherbh_dx,
      count(distinct case when bh_conditions ilike '%personality%' then claim_from_dt end) as mh_personality_dx,
      count(distinct case when bh_conditions ilike '%ptsd%' then claim_from_dt end) as mh_ptsd_dx,
      count(distinct case when dx_primary ilike 'G47%' or dx_secondary ilike '%G47%' or chronic_conditions ilike '%sleep%'
       then claim_from_dt end)     as sleep_dos_chronic
       ,count(distinct case when chronic_conditions ilike '%pain%' or dx_primary ilike 'R52%' or dx_secondary ilike '%R52%'
       then claim_from_dt end)     as pain_dos_chronic
       ,count(distinct case when chronic_conditions ilike '%cancer%'
       then claim_from_dt end)     as cancer_dos_chronic
       ,count(distinct case when dx_primary ilike 'R4585%' or dx_secondary ilike '%R4585%'
       then claim_from_dt end)     as mh_suicide_homicide_ideation_dx
       ,count(distinct case when dx_primary ilike 'T1491%' or dx_secondary ilike '%T1491%'
       then claim_from_dt end)     as mh_suicide_attempt_injury_dx
       ,count(distinct case when ccs ilike '[102,%' or ccs ilike '% 102,%' or ccs ilike '[102]' or ccs ilike '%, 102]'
       then claim_from_dt end)     as chestpain_unspecified_dos
      ,count(distinct case when ccs ilike '[250,%' or ccs ilike '% 250,%' or ccs ilike '[250]' or ccs ilike '%, 250]'
       then claim_from_dt end)     as nausea_dos
      ,count(distinct case when ccs ilike '[251,%' or ccs ilike '% 251,%' or ccs ilike '[251]' or ccs ilike '%, 251]'
       then claim_from_dt end)     as abdominal_pain_dos
      ,count(distinct case when ccs ilike '[252,%' or ccs ilike '% 252,%' or ccs ilike '[252]' or ccs ilike '%, 252]'
       then claim_from_dt end)     as fatigue_dos
      ,count(distinct case when is_bhp then claim_from_dt end) as bhp_visits
      ,count(distinct case when is_pcp then claim_from_dt end) as pcp_visits
      ,count(distinct case when is_specialist and not is_bhp then perf_npi end) as ct_specialists_seen
      ,sum(paid_amt) as total_medical_cost
      ,count(distinct case when ccs ilike '[84,%' or ccs ilike '% 84,%' or ccs ilike '[84]' or ccs ilike '%, 84]'
       then claim_from_dt end)     as headache_dos_chronic
      ,count(distinct case when ccs ilike '[88,%' or ccs ilike '% 88,%' or ccs ilike '[88]' or ccs ilike '%, 88]'
       or ccs ilike '[89,%' or ccs ilike '% 89,%' or ccs ilike '[89]' or ccs ilike '%, 89]'
       then claim_from_dt end)     as glaucoma_vision_dos_chronic
       ,count(distinct case when ccs ilike '[98,%' or ccs ilike '% 98,%' or ccs ilike '[98]' or ccs ilike '%, 98]'
       or ccs ilike '[99,%' or ccs ilike '% 99,%' or ccs ilike '[99]' or ccs ilike '%, 99]'
       then claim_from_dt end)     as hypertension_dos_chronic
       ,count(distinct case when ccs ilike '[106,%' or ccs ilike '% 106,%' or ccs ilike '[106]' or ccs ilike '%, 106]'
       then claim_from_dt end)     as dysrhythmias_dos_chronic
       ,count(distinct case when ccs ilike '[127,%' or ccs ilike '% 127,%' or ccs ilike '[127]' or ccs ilike '%, 127]'
       then claim_from_dt end)     as copd_dos_chronic
      ,count(distinct case when ccs ilike '[128,%' or ccs ilike '% 128,%' or ccs ilike '[128]' or ccs ilike '%, 128]'
       then claim_from_dt end)     as asthma_dos_chronic
      ,count(distinct case when ccs ilike '[49,%' or ccs ilike '% 49,%' or ccs ilike '[49]' or ccs ilike '%, 49]'
       or ccs ilike '[50,%' or ccs ilike '% 50,%' or ccs ilike '[50]' or ccs ilike '%, 50]'
       then claim_from_dt end)     as diabetes_dos_chronic
       ,count(distinct case when dx_primary ilike 'J00%' or dx_secondary ilike '%J00%' or
       dx_primary ilike 'J01%' or dx_secondary ilike '%J01%'
       then claim_from_dt end)     as common_cold_sinusitis_dos
       ,count(distinct case when dx_primary ilike 'K50%' or dx_secondary ilike '%K50%' or
       dx_primary ilike 'K51%' or dx_secondary ilike '%K51%' or
       dx_primary ilike 'K52%' or dx_secondary ilike '%K52%' or
       dx_primary ilike 'K58%' or dx_secondary ilike '%K58%'
       then claim_from_dt end)     as crohns_ibs_dos_chronic
       ,count(distinct case when ccs ilike '[171,%' or ccs ilike '% 171,%' or ccs ilike '[171]' or ccs ilike '%, 171]'
       then claim_from_dt end)     as menstrual_disorders_dos
       ,count(distinct case when dx_primary ilike 'L40%' or dx_secondary ilike '%L40%' or
       dx_primary ilike 'L41%' or dx_secondary ilike '%L41%' or
       dx_primary ilike 'L2%' or dx_secondary ilike '%L2%' or
       dx_primary ilike 'L30%' or dx_secondary ilike '%L30%' or
       dx_primary ilike 'L70%' or dx_secondary ilike '%L70%' or
       dx_primary ilike 'L71%' or dx_secondary ilike '%L71%'
       then claim_from_dt end)     as psoriasis_eczema_acne_dos
       ,count(distinct case when ccs ilike '[100,%' or ccs ilike '% 100,%' or ccs ilike '[100]' or ccs ilike '%, 100]'
       or ccs ilike '[101,%' or ccs ilike '% 101,%' or ccs ilike '[101]' or ccs ilike '%, 101]'
       or ccs ilike '[103,%' or ccs ilike '% 103,%' or ccs ilike '[103]' or ccs ilike '%, 103]'
       or ccs ilike '[104,%' or ccs ilike '% 104,%' or ccs ilike '[104]' or ccs ilike '%, 104]'
       then claim_from_dt end)     as heart_disease_mi_dos_chronic
    """ + """
    from members left join bcbs_nc_prod.medical_claim med on members.member_id = med.member_id
        where claim_from_dt < '2019-05-01' and claim_from_dt >= '2018-05-01'
        group by 1
      ),
    """

    rx_query = """
    rx as (
        select
        members.member_id member_id,
        count(distinct case when ref.is_bh then fill_dt end) as mh_fills
        ,min(case when ref.is_bh then datediff(day, fill_dt, '2019-05-01') else 365 end) as days_since_mh_fill
        ,sum(paid_amt) as total_rx_cost
        ,count(distinct case when ref.is_bh then gen_drug_name end) as bh_drug_ct
        ,count(distinct case when ref.is_bh then coalesce(drug_strength::text, regexp_substr(drug_name, ' [^mg]*')) end) as ct_bh_dosages
        ,count(distinct case when ref.is_bh then bh_cat_broad end) as bh_drug_cat_ct
        ,count(distinct case when ref.is_chronic then fill_dt end) as chronic_fills
        ,count(distinct case when ref.is_chronic then gen_drug_name end) as chronic_drug_ct
        ,count(distinct case when ref.is_chronic then rx.therapeutic_class_broad end) as chronic_therapy_class_ct
        ,count(distinct case when is_insomnia_sh then fill_dt end) as insomnia_fills
        ,count(distinct case when is_insomnia_sh then gen_drug_name end) as insomnia_drug_ct
      -- types of bh drugs
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Antidepressant_SSRI' then fill_dt end) as ssri_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Anxiolytic_Benzodiazepine' then fill_dt end) as benzo_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Antidepressant_Misc' then fill_dt end) as antidepmisc_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Antidepressant_TCA' then fill_dt end) as antideptca_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Antipsychotic_Second_Generation' then fill_dt end) as antipsych2nd_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Stimulant_Amphetamine' then fill_dt end)
        as stimulantamphet_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Stimulant_Methylphenidate_Deriv' then fill_dt end)
        as stimulantmethyl_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Anxiolytic_nonBenzodiazepine' then fill_dt end)
        as anxiolyticsnonbenzo_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'MoodStabilizer' then fill_dt end)
        as moodstab_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Antipsychotic_First_Generation' then fill_dt end)
        as antipsychotic1st_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Stimulant_ADHD' then fill_dt end)
        as stimulantadhd_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'nonStimulant_ADHD' then fill_dt end)
        as nonstimulantadhd_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'AlcoholDeterrent' then fill_dt end)
        as alcdeter_fills
        ,count(distinct case when ref.is_bh and ref.bh_cat_fine ilike 'Antidepressant_MAOI' then fill_dt end)
        as antidepmaoi_fills
    """ + drugs_strings + """
    from members left join bcbs_nc_prod.rx_claim rx on members.member_id = rx.member_id
    left join (
    select distinct ndc11, therapeutic_class_broad, is_bh::boolean, bh_cat_fine, is_chronic::boolean
    from reference_tables.gsddb_ndc11_reference
    ) ref on ref.ndc11 = rx.ndc
        where fill_dt >= '2018-05-01' and fill_dt < '2019-05-01'
        group by 1
      )
    """

    base_query_med_rx = med_query + rx_query

    inference_members_query = """
    with
    members as (
        select mem.member_id member_id,
          member_age,
          member_is_male is_male,
          member_zip,
          member_region,
          case when member_subscriber_relationship = 'spouse_partner' or
          member_marital_status = 'Married' then TRUE else FALSE end as is_married_partner,
          case when member_marital_status = 'Divorced' then TRUE else FALSE end as is_divorced,
          case when member_marital_status = 'Widowed' then TRUE else FALSE end as is_widowed,
          subscriber_id,
          longitude,
          latitude
        from bcbs_nc_prod.member_month mem
        left join (
            select member_id
            from bcbs_nc_prod.exclusions
            where "month" = '2019-05-01' and (
              age_excluded or ltc_excluded or dementia_excluded or hospice_excluded
              or capitation_excluded or coma_excluded or esrd_excluded
            )
            ) e on mem.member_id = e.member_id
        left join (
            select member_id
            from bcbs_nc_prod.prior_bhp
            where "month" = '2019-05-01' and (is_in_care_bhp or is_in_care_prescriber)
            ) p on mem.member_id = p.member_id
        left join (
            select postal_code, latitude, longitude
            from reference_tables.zip_codes
        ) z on mem.member_zip = z.postal_code
        where is_eligible and is_treatment and e.member_id is null and p.member_id is null
        and "month" = '2019-05-01' --and member_state = 'NC'
      )
    """

    training_members_query = """
    with
    members as (
    select mem.member_id member_id,
          member_age,
          member_is_male is_male,
          member_zip,
          member_region,
          case when member_subscriber_relationship = 'spouse_partner' or
          member_marital_status = 'Married' then TRUE else FALSE end as is_married_partner,
          case when member_marital_status = 'Divorced' then TRUE else FALSE end as is_divorced,
          case when member_marital_status = 'Widowed' then TRUE else FALSE end as is_widowed,
          subscriber_id,
          longitude,
          latitude
        from bcbs_nc_prod.member_month mem
        left join (
            select member_id
            from bcbs_nc_prod.exclusions
            where "month" = '2018-01-01' and (
              age_excluded or ltc_excluded or dementia_excluded or hospice_excluded
              or capitation_excluded or coma_excluded or esrd_excluded
            )
            ) e on mem.member_id = e.member_id
        left join (
            select member_id
            from bcbs_nc_prod.prior_bhp
            where "month" = '2018-01-01' and (is_in_care_bhp or is_in_care_prescriber)
            ) p on mem.member_id = p.member_id
        left join (
        select member_id, count("month") ct_eligible
        from bcbs_nc_prod.member_month
        where is_eligible and is_treatment and is_rx_benefit and "month" > '2017-01-01'
        and "month" <= '2019-01-01'
        group by 1
        having count("month") = 24
        ) elig on elig.member_id = mem.member_id
        left join (
            select postal_code, latitude, longitude
            from reference_tables.zip_codes
        ) z on mem.member_zip = z.postal_code
        where is_eligible and is_treatment and e.member_id is null and p.member_id is null
        and "month" = '2018-01-01' and elig.member_id is not null --and member_state = 'NC'
    )
    """

    response_query = """
    ,response as (
        select members.member_id member_id,
          coalesce(ip_er_visit_response, false) as ip_er_visit_response,
          coalesce(
        case when ip_er_bh_visit or mh_adhd_dx_2018 > 1 or mh_anxiety_dx_2018 > 1
        or mh_conduct_dx_2018 > 1 or mh_depression_dx_2018 > 1 or
        mh_eating_dx_2018 > 1 or mh_ocd_dx_2018 > 1 or mh_otherbh_dx_2018 > 1 or mh_personality_dx_2018 > 1
        or mh_ptsd_dx_2018 > 1 or mh_fills > 2 
        or mh_bipolar_dx_2018 > 1 or mh_schizo_dx_2018 > 1
        or mh_suicide_homicide_ideation_dx_2018 > 1 or mh_suicide_attempt_injury_dx_2018 > 1
        then true else false end,
        false
      ) as has_bh_response,
      coalesce(medical_paid_2018, 0.0) + coalesce(rx_paid_2018, 0.0) as total_paid_response
        from members left join (
          select member_id,
            bool_or(case when svc_cat in ('04_OP_Visit_ER', '01_IP_Acute') then true else false end) as ip_er_visit_response,
            bool_or(distinct case when svc_cat in ('04_OP_Visit_ER', '01_IP_Acute') and bh_conds_cnt > 0 then true else false end) as ip_er_bh_visit,
          count(distinct case when bh_conditions ilike '%bipolar%' then claim_from_dt end) as mh_bipolar_dx_2018,
          count(distinct case when bh_conditions ilike '%schizophrenia%' then claim_from_dt end) as mh_schizo_dx_2018,
          count(distinct case when bh_conditions ilike '%adhd%' then claim_from_dt end) as mh_adhd_dx_2018,
          count(distinct case when bh_conditions ilike '%anxiety%' then claim_from_dt end) as mh_anxiety_dx_2018,
          count(distinct case when bh_conditions ilike '%conduct%' then claim_from_dt end) as mh_conduct_dx_2018,
          count(distinct case when bh_conditions ilike '%depression%' then claim_from_dt end) as mh_depression_dx_2018,
          count(distinct case when bh_conditions ilike '%eating%' then claim_from_dt end) as mh_eating_dx_2018,
          count(distinct case when bh_conditions ilike '%ocd%' then claim_from_dt end) as mh_ocd_dx_2018,
          count(distinct case when bh_conditions ilike '%otherbh%' then claim_from_dt end) as mh_otherbh_dx_2018,
          count(distinct case when bh_conditions ilike '%personality%' then claim_from_dt end) as mh_personality_dx_2018,
          count(distinct case when bh_conditions ilike '%ptsd%' then claim_from_dt end) as mh_ptsd_dx_2018,
          sum(paid_amt) as medical_paid_2018
          ,count(distinct case when dx_primary ilike 'R4585%' or dx_secondary ilike '%R4585%'
          then claim_from_dt end)     as mh_suicide_homicide_ideation_dx_2018
          ,count(distinct case when dx_primary ilike 'T1491%' or dx_secondary ilike '%T1491%'
          then claim_from_dt end)     as mh_suicide_attempt_injury_dx_2018
          from bcbs_nc_prod.medical_claim
          where claim_from_dt >= '2018-01-01' and claim_from_dt < '2019-01-01'
          group by 1
          ) med on members.member_id = med.member_id
        left join (
            select member_id,
              count(distinct case when ref.is_bh then fill_dt end) as mh_fills,
              sum(paid_amt) as rx_paid_2018
            from bcbs_nc_prod.rx_claim rx left join
            (
            select distinct ndc11, therapeutic_class_broad, is_bh::boolean, bh_cat_fine
            from reference_tables.gsddb_ndc11_reference
            ) ref on ref.ndc11 = rx.ndc
            where fill_dt >= '2018-01-01' and fill_dt < '2019-01-01'
            group by 1
            ) rx on members.member_id = rx.member_id
      )
    """

    # family size & family members with MH
    family_query = """
    ,family_pairs as (
    select m1.member_id as left_member_id,
    m1.subscriber_id subscriber_id,
    m2.member_id as right_member_id
    from bcbs_nc_prod.member_month m1 left join bcbs_nc_prod.member_month m2
    on m1.subscriber_id = m2.subscriber_id and m1.member_id != m2.member_id
    where m1."month" = '2019-05-01' and m2."month" = '2019-05-01'
    )

    ,family_size as (
    select left_member_id as member_id,
    count(subscriber_id) as ct_family
    from family_pairs
    group by 1
    )

    ,mh_presence as (
    select members.member_id member_id,
    coalesce(
        case when ip_er_bh_visit_count > 0 or mh_adhd_dx > 1 or mh_anxiety_dx > 1
        or mh_conduct_dx > 1 or mh_depression_dx > 1 or
        mh_eating_dx > 1 or mh_ocd_dx > 1 or mh_otherbh_dx > 1 or mh_personality_dx > 1
        or mh_ptsd_dx > 1 or mh_fills > 2 
        or mh_bipolar_dx > 1 or mh_schizo_dx > 1
        or mh_suicide_homicide_ideation_dx > 1 or mh_suicide_attempt_injury_dx > 1
        then true else false end,
        false
      ) as has_bh
    from members left join medical on members.member_id = medical.member_id
    left join rx on members.member_id = rx.member_id
    )

    ,mh_family as (
    select left_member_id member_id,
    count(case when has_bh then TRUE end) as family_has_mh
    from family_pairs f left join mh_presence mh on f.right_member_id = mh.member_id
    group by 1
    )

    """

    training_query = """
    select *
    from response left join members on response.member_id = members.member_id
    left join medical on response.member_id = medical.member_id
    left join rx on response.member_id = rx.member_id
    left join family_size on response.member_id = family_size.member_id
    left join mh_presence on response.member_id = mh_presence.member_id
    left join mh_family on response.member_id = mh_family.member_id
    """
    inference_query = """
    select *
    from members left join medical on members.member_id = medical.member_id
    left join rx on members.member_id = rx.member_id
    left join family_size on members.member_id = family_size.member_id
    left join mh_presence on members.member_id = mh_presence.member_id
    left join mh_family on members.member_id = mh_family.member_id
    """ 
    
    TRAINING_QUERY = (
        training_members_query.replace('2018-01-01', train_end).replace('2017-01-01', train_start).replace('2019-01-01', response_end) + (
            base_query_med_rx
            .replace('2018-05-01', train_start)
            .replace('2019-05-01', train_end)
            .replace('%', '%%') # for annoying python print formatting reasons
        ) +
        response_query.replace('%', '%%').replace('2018-01-01', train_end).replace('2019-01-01', response_end) + 
        family_query.replace('2019-05-01', train_end) +
        training_query
    )
    INFERENCE_QUERY = (
        inference_members_query + 
        base_query_med_rx.replace('%', '%%') +
        family_query +
        inference_query
    )
    # replace with customer name
    TRAINING_QUERY = TRAINING_QUERY.replace("bcbs_nc_prod", customer)
    INFERENCE_QUERY = INFERENCE_QUERY.replace("bcbs_nc_prod", customer)
    return TRAINING_QUERY, INFERENCE_QUERY
