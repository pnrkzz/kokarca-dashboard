import { useMemo, useState, type ReactNode } from "react";

import localCollectionUi from "../assets/methodology/localCollectionUi.png";
import nationalCollectionUi from "../assets/methodology/nationalCollectionUi.png";
import youtubeCollectionUi from "../assets/methodology/youtubeCollectionUi.png";
import gptAutomationSetup from "../assets/methodology/gptAutomationSetup.png";
import gptAutomationMapping from "../assets/methodology/gptAutomationMapping.png";

/* =========================================================
   TYPES
   ========================================================= */

type SectionKey =
  | "data"
  | "llm"
  | "climate"
  | "geovisual";

type DetailKey =
  | "data-overview"
  | "collection-automation"
  | "national"
  | "local"
  | "youtube-collect"
  | "official"
  | "academic"
  | "corpus"

  | "llm-overview"
  | "youtube-preprocess"
  | "prefilter"
  | "prompt"
  | "extraction"
  | "postprocess"
  | "automation"

  | "climate-overview"
  | "climategpt-eval"
  | "climategpt-research"
  | "era5"
  | "consistency"

  | "geo-overview"
  | "map"
  | "timeline"
  | "documents"
  | "climate-view"
  | "interventions"
  | "stc"
  | "questions";


type NavItem = {
  key: DetailKey;
  icon: string;
  label: string;
  subtitle?: string;
};


type PreviewImage = {
  src: string;
  title?: string;
  caption?: string;
};

type PreviewData = {
  title: string;

  kicker?: string;

  image?: string;
  images?: PreviewImage[];

  description?: string;

  features?: string[];

  output?: string;

  contentTitle?: string;
  content?: string;

  contentLanguage?: "text" | "json";
};
/* =========================================================
   NAVIGATION
   ========================================================= */

const SECTIONS: {
  key: SectionKey;
  icon: string;
  label: string;
  items: NavItem[];
}[] = [

  {
    key: "data",
    icon: "📂",
    label: "Veri Üretim Altyapısı",

    items: [
      {
        key: "data-overview",
        icon: "◎",
        label: "Genel Akış",
      },

      {
        key: "collection-automation",
        icon: "⚙",
        label: "Veri Toplama Otomasyonları",
        subtitle: "Form + butonlu arayüz",
      },

      {
        key: "national",
        icon: "📰",
        label: "Ulusal Haber",
        subtitle: "Veri toplama otomasyonu",
      },

      {
        key: "local",
        icon: "⌖",
        label: "Yerel Haber",
        subtitle: "Veri toplama otomasyonu",
      },

      {
        key: "youtube-collect",
        icon: "▶",
        label: "YouTube",
        subtitle: "Video + transkript",
      },

      {
        key: "official",
        icon: "🏛",
        label: "Resmî Kurum",
        subtitle: "Kurumsal kaynaklar",
      },

      {
        key: "academic",
        icon: "🎓",
        label: "Akademik",
        subtitle: "Çoklu akademik kaynak",
      },

      {
        key: "corpus",
        icon: "📚",
        label: "Analiz Korpusu",
        subtitle: "2036 belge",
      },
    ],
  },


  {
    key: "llm",
    icon: "🤖",
    label: "LLM ile Bilgi Çıkarımı",

    items: [
      {
        key: "llm-overview",
        icon: "◎",
        label: "Genel Akış",
      },

      {
        key: "youtube-preprocess",
        icon: "🧹",
        label: "Kaynak Bazlı Ön İşleme",
        subtitle: "YouTube'a özel",
      },

      {
        key: "prefilter",
        icon: "🔎",
        label: "Hedef Tür Ön Filtresi",
      },

      {
        key: "prompt",
        icon: "💬",
        label: "Prompt Tasarımı",
        subtitle: "System + User + Schema",
      },

      {
        key: "extraction",
        icon: "⚙",
        label: "GPT Bilgi Çıkarımı",
        subtitle: "GPT-4.1-mini",
      },

      {
        key: "postprocess",
        icon: "🛡",
        label: "Kalite ve Güvenlik",
        subtitle: "Lokasyon + iklim kontrolleri",
      },

      {
        key: "automation",
        icon: "🔁",
        label: "Genellenebilir Otomasyon",
        subtitle: "Yeniden kullanılabilir çıkarım",
      },
    ],
  },


  {
    key: "climate",
    icon: "🌍",
    label: "İklim ve İklim Değişikliği Analizi",

    items: [
      {
        key: "climate-overview",
        icon: "◎",
        label: "Genel Akış",
      },

      {
        key: "climategpt-eval",
        icon: "🧠",
        label: "ClimateGPT Değerlendirmesi",
        subtitle: "RunPod + web arayüzü",
      },

      {
        key: "climategpt-research",
        icon: "🧾",
        label: "ClimateGPT Araştırma Kullanımı",
        subtitle: "Narrative + evidence",
      },

      {
        key: "era5",
        icon: "🌡",
        label: "ERA5",
        subtitle: "Gözlemsel iklim karşılaştırması",
      },

      {
        key: "consistency",
        icon: "✅",
        label: "Tutarlılık Analizi",
      },
    ],
  },


  {
    key: "geovisual",
    icon: "🗺",
    label: "Bağlantılı Jeogörsel Analiz",

    items: [
      {
        key: "geo-overview",
        icon: "◎",
        label: "Genel Akış",
      },

      {
        key: "map",
        icon: "🗺",
        label: "Harita",
      },

      {
        key: "timeline",
        icon: "◷",
        label: "Zaman Görünümü",
      },

      {
        key: "documents",
        icon: "📄",
        label: "Belgeler",
      },

      {
        key: "climate-view",
        icon: "🌦",
        label: "İklim",
      },

      {
        key: "interventions",
        icon: "🏛",
        label: "Müdahaleler",
      },

      {
        key: "stc",
        icon: "🧊",
        label: "STC",
      },

      {
        key: "questions",
        icon: "?",
        label: "Soru-Cevap",
      },
    ],
  },
];


/* =========================================================
   EXTRACTION FIELDS
   ========================================================= */

const extractionFields = [
  "summary_tr",
  "location_mentions / provinces_raw",
  "spread_observation_year",
  "intervention_year",
  "crops_raw",
  "event_type_raw",
  "government_action_raw",
  "institution_raw",
  "climate_signal_raw",
  "climate_factor_type_raw",
  "climate_comment",
  "trend_direction",
  "extraction_confidence",
  "evidence_sentence",
];


/* =========================================================
   REUSABLE COMPONENTS
   ========================================================= */

function IOGrid({
  purpose,
  input,
  process,
  output,
}: {
  purpose: ReactNode;
  input: ReactNode;
  process: ReactNode;
  output: ReactNode;
}) {

  return (
    <div className="research-io-grid">

      <article>
        <span>AMAÇ</span>
        <div>{purpose}</div>
      </article>

      <article>
        <span>GİRDİ</span>
        <div>{input}</div>
      </article>

      <article>
        <span>İŞLEM</span>
        <div>{process}</div>
      </article>

      <article>
        <span>ÇIKTI</span>
        <div>{output}</div>
      </article>

    </div>
  );
}


function Flow({
  items,
}: {
  items: string[];
}) {

  return (
    <div className="research-flow">

      {items.map((item, i) => (

        <div
          className="research-flow-fragment"
          key={`${item}-${i}`}
        >

          <div className="research-flow-node">
            {item}
          </div>

          {i < items.length - 1 && (
            <div className="research-flow-arrow">
              →
            </div>
          )}

        </div>

      ))}

    </div>
  );
}


/* =========================================================
   PREVIEW BUTTON
   ========================================================= */

function PreviewButton({
  onClick,
  label = "Arayüz",
  icon = "▣",
}: {
  onClick: () => void;
  label?: string;
  icon?: string;
}) {
  return (
    <button
      type="button"
      className="method-preview-button"
      onClick={onClick}
      title={`${label} detayını görüntüle`}
    >
      <span>{icon}</span>
      <span>{label}</span>
    </button>
  );
}
/* =========================================================
   COLLECTION CARD
   ========================================================= */

function CollectionCard({
  icon,
  title,
  subtitle,
  image,
  description,
  features,
  output,
  onPreview,
}: {
  icon: string;
  title: string;
  subtitle: string;

  image?: string;
  description?: string;
  features?: string[];
  output?: string;

  onPreview?: (data: PreviewData) => void;
}) {

  const hasPreview = Boolean(
    image &&
    description &&
    features &&
    output &&
    onPreview
  );


  function openPreview() {

    if (
      !image ||
      !description ||
      !features ||
      !output ||
      !onPreview
    ) {
      return;
    }

    onPreview({
      title,
      image,
      description,
      features,
      output,
    });
  }


  return (
    <article className="collection-automation-card">

      <div className="collection-card-head">

        <span className="collection-card-icon">
          {icon}
        </span>

        <div className="collection-card-copy">

          <b>
            {title}
          </b>

          <small>
            {subtitle}
          </small>

        </div>


        {hasPreview && (
          <PreviewButton
            onClick={openPreview}
          />
        )}

      </div>


      <div
        className={
          hasPreview
            ? "collection-ui-badge"
            : "collection-script-badge"
        }
      >

        {hasPreview
          ? "Kullanıcı arayüzü"
          : "Script / yarı otomatik"}

      </div>

    </article>
  );
}


/* =========================================================
   DETAIL VIEW
   ========================================================= */

function Detail({
  detail,
  onPreview,
}: {
  detail: DetailKey;
  onPreview: (data: PreviewData) => void;
}) {


  /* =======================================================
     DATA — OVERVIEW
     ======================================================= */


  /* FORM_TABANLI_V22_METODOLOJI_2026 */
  if (detail === "data-overview" || detail === "collection-automation" ||
      detail === "llm-overview" || detail === "automation") {
    const isData = detail === "data-overview";
    const isCollect = detail === "collection-automation";
    const isLlm = detail === "llm-overview";
    const heading = isData ? "Form Tabanlı Veri Üretim Altyapısı" :
      isCollect ? "Beş Kaynak İçin Veri Toplama Otomasyonları" :
      isLlm ? "İki Aşamalı GPT Bilgi Çıkarımı" :
      "Genellenebilir GPT-1 ve GPT-2 Otomasyonu";
    const steps: [string,string][] = isCollect ? [
      ["Ulusal haber — 601 belge","Kaynak/domain, anahtar kelime, URL ve tam metin toplama"],
      ["Yerel haber — 583 belge","Yerel kaynaklar, il/ilçe sorguları, tam metin"],
      ["YouTube — 302 kayıt","Video keşfi, transkript ve içerik filtresi"],
      ["Akademik — 294 belge","Bibliyografik arama, metadata, erişilebilir tam metin/PDF"],
      ["Resmî kurum — 256 belge","Kurum sayfaları, açıklamalar ve kaynak kayıtları"]
    ] : isLlm || detail === "automation" ? [
      ["Form ayarları","Proje, hedef tür, bilimsel ad/eş adlar, dönem, CSV ve kolon eşleştirme"],
      ["GPT-1 · GPT-4.1-mini","Bağımsız olay, yer, zaman, ürün, zarar, müdahale, kurum, iklim ve doğrudan kanıt çıkarımı"],
      ["GPT-2 · GPT-4.1","Aday olayları sınıflandırma, ayrıştırma, ilişkilendirme ve kanıt tutarlılığını değerlendirme"],
      ["Doğrulama","Belge–olay kimliği, zaman/konum ve kanıt desteği denetimleri"],
      ["Yapılandırılmış çıktı","JSON, V22 olay ve ilişkisel analiz tabloları"]
    ] : [
      ["1 · Form tabanlı toplama","Ulusal haber, yerel haber, YouTube, akademik ve resmî kaynaklar"],
      ["2 · Analiz korpusu","2.036 belge; özgün kaynak, kategori, kimlik ve metin bağlantısı korunur"],
      ["3 · GPT-1","GPT-4.1-mini ile belge içindeki bağımsız olayları ve ilişkili alanları çıkarma"],
      ["4 · GPT-2","GPT-4.1 ile olay sınıflandırma, yapılandırma ve kanıt ilişkilerini değerlendirme"],
      ["5 · Kalite ve saklama","V22 doğrulama; JSON, ilişkisel tablolar ve MongoDB veri yönetimi katmanı"],
      ["6 · Jeogörsel analiz","FastAPI, React/MapLibre; olay haritası, STC, iklim, müdahale ve belgeler"]
    ];
    return <>
      <h2>{heading}</h2>
      <p className="research-lead">
        {isCollect
          ? "Veri elde etme süreçleri, araştırmacının parametreleri formlarla yönettiği kaynak-özel otomasyonlara dönüştürülmüştür."
          : isLlm || detail === "automation"
          ? "İki GPT aşaması, proje ve tür ayarları ile veri yükleme/kolon eşleştirme formlarından yönetilen otomasyonun parçalarıdır. GPT-1 aday olayları çıkarır; GPT-2 olayları sınıflandırıp yapılandırır."
          : "Çok kaynaklı veri toplama ve iki aşamalı GPT analizi form tabanlı otomasyonlarla yürütülür. Bu metodoloji ekranı, araştırmada kullanılan mimariyi tanıtır; buradaki açıklamalar işlemleri yeniden çalıştırmaz."}
      </p>
      <div style={{display:"grid",gap:12,marginTop:16}}>
        {steps.map(([name,desc]) => <article key={name} style={{padding:16,border:"1px solid rgba(145,155,190,.35)",borderRadius:12}}>
          <h3 style={{margin:"0 0 7px"}}>{name}</h3>
          <p style={{margin:0,lineHeight:1.6}}>{desc}</p>
        </article>)}
      </div>
      {(isCollect || isData) && <>
        <h3>Gerçek veri toplama arayüzleri</h3>
        <div className="collection-automation-grid">
          <CollectionCard icon="📰" title="Ulusal haber" subtitle="Formlu otomasyon" image={nationalCollectionUi}
            description="Ulusal haber toplama ekranı" features={["Arama ve kaynak ayarları","URL keşfi","Tam metin"]} output="601 belge" onPreview={onPreview}/>
          <CollectionCard icon="📍" title="Yerel haber" subtitle="Formlu otomasyon" image={localCollectionUi}
            description="Yerel haber toplama ekranı" features={["Yerel kaynaklar","Anahtar kelimeler","Tam metin"]} output="583 belge" onPreview={onPreview}/>
          <CollectionCard icon="▶" title="YouTube" subtitle="Formlu otomasyon" image={youtubeCollectionUi}
            description="Video ve transkript toplama ekranı" features={["Video arama","Transkript","Filtreleme"]} output="302 kayıt" onPreview={onPreview}/>
        </div>
        <p>Akademik ve resmî kaynakların yeni form ekranları henüz bu bileşende görsel olarak tanımlı değildir.</p>
      </>}
      {(isLlm || detail === "automation" || isData) && <>
        <h3>Gerçek GPT otomasyon arayüzleri</h3>
        <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
          <PreviewButton label="Proje ve tür ayarları" onClick={() => onPreview({
            title:"Proje ve tür formu",image:gptAutomationSetup,
            description:"Proje, hedef tür ve dönem ayarları",features:["Proje","Hedef tür","Dönem"],output:"Otomasyon ayarları"
          })}/>
          <PreviewButton label="CSV ve kolon eşleştirme" onClick={() => onPreview({
            title:"CSV ve kolon eşleştirme",image:gptAutomationMapping,
            description:"Veri dosyası ve alan eşleştirme ekranı",features:["CSV","Metin / tarih / URL alanları","Test modu"],output:"Analiz girdisi"
          })}/>
        </div>
        <p>Bu iki görsel otomasyonun kurulum ekranlarını gösterir; GPT-1 ve GPT-2 için ayrı çalışma ekranları mevcut görseller arasında yer almamaktadır.</p>
      </>}
      <IOGrid purpose="Kaynağı izlenebilir, tekrar üretilebilir olay tabanlı GeoAI analizi."
        input="Kaynak kimlikli belgeler, araştırmacı form parametreleri ve veri alanları."
        process="Toplama → GPT-1 → GPT-2 → kanıt/konum/zaman kontrolü → sürümlü veri."
        output="JSON, V22 ilişkisel tablolar, MongoDB veri katmanı ve bağlantılı jeogörsel analizler."/>
      <p>MongoDB, veri yönetimi mimarisinin bir bileşenidir; bu tanıtım ekranı canlı veritabanı bağlantısını doğrulamaz. Kanıt desteği, müdahalenin sahada bağımsız olarak doğrulandığı anlamına gelmez.</p>
    </>;
  }

  if (false && detail === "data-overview") {

    return (
      <>

        <h2>
          Veri Üretim Altyapısı
        </h2>

        <p className="research-lead">
          Kahverengi kokarca ile ilgili bilgi farklı web ve akademik
          kaynaklara dağılmış durumdaydı. Bu nedenle araştırma,
          hazır bir veri setiyle değil; farklı kaynaklardan belge
          üreten veri toplama iş akışlarıyla başladı.
        </p>


        <Flow
          items={[
            "Arama stratejisi",
            "Kaynak-özel otomasyonlar",
            "Tam metin / transkript",
            "Filtreleme ve kalite kontrol",
            "2036 belgelik analiz korpusu",
          ]}
        />


        <div className="source-counts">

          <div>
            <b>601</b>
            <span>Ulusal haber</span>
          </div>

          <div>
            <b>583</b>
            <span>Yerel haber</span>
          </div>

          <div>
            <b>302</b>
            <span>YouTube</span>
          </div>

          <div>
            <b>294</b>
            <span>Akademik</span>
          </div>

          <div>
            <b>256</b>
            <span>Resmî</span>
          </div>

        </div>


        <IOGrid
          purpose="Dağınık çevresel bilgiyi sistematik ve tekrar üretilebilir bir analiz korpusunda toplamak."
          input="Anahtar kelimeler, hedef tür, kaynak listeleri ve web/akademik veri kaynakları."
          process="Kaynak türüne göre ayrı veri toplama iş akışları ve kalite kontrolleri."
          output={
            <>
              <strong>2036 belge</strong> içeren çok kaynaklı analiz korpusu.
            </>
          }
        />

      </>
    );
  }



  /* =======================================================
     DATA — COLLECTION AUTOMATION
     ======================================================= */

  if (false && detail === "collection-automation") {

    return (
      <>

        <h2>
          Veri Toplama Otomasyonları
        </h2>


        <p className="research-lead">
          Veri toplama yalnızca sabit kodlarla yürütülmedi.
          Ulusal haber, yerel haber ve YouTube kaynakları için
          araştırmacının hedef tür, kaynak, anahtar kelime ve
          toplama parametrelerini belirleyebildiği butonlu
          kullanıcı arayüzleri geliştirildi. Akademik ve resmî
          kaynaklarda ise aynı türde bir form arayüzü kullanılmadı;
          bu kaynaklar script ve yarı otomatik toplama iş akışlarıyla
          yürütüldü.
        </p>


        <div className="collection-automation-grid">


          <CollectionCard
            icon="📰"
            title="Ulusal Haber"
            subtitle="Ulusal medya veri toplama otomasyonu"
            image={nationalCollectionUi}
            description="Ulusal haber kaynaklarında hedef tür ile ilişkili içeriklerin sistematik olarak toplanması."
            features={[
              "Tür adı ve anahtar kelimeler",
              "Domain listesi",
              "Sitemap taraması",
              "URL keşfi",
              "Full text çıkarımı",
              "Batch işlemleri",
            ]}
            output="601 belge"
            onPreview={onPreview}
          />


          <CollectionCard
            icon="⌖"
            title="Yerel Haber"
            subtitle="Yerel medya veri toplama otomasyonu"
            image={localCollectionUi}
            description="Yerel haber kaynaklarında il ve ilçe düzeyindeki gözlem, yayılım, zarar ve mücadele bilgilerinin sistematik olarak toplanması."
            features={[
              "Tür adı",
              "Ek anahtar kelimeler",
              "Yerel domain listesi",
              "URL keşfi",
              "Full text çıkarımı",
              "Batch işlemleri",
            ]}
            output="583 belge"
            onPreview={onPreview}
          />


          <CollectionCard
            icon="▶"
            title="YouTube"
            subtitle="Video ve transkript veri toplama otomasyonu"
            image={youtubeCollectionUi}
            description="YouTube videolarındaki saha anlatıları, uzman açıklamaları ve yerel gözlemlerin transkript üzerinden metinsel veriye dönüştürülmesi."
            features={[
              "İstilacı tür sorgusu",
              "Sayfa sayısı",
              "Minimum metin uzunluğu",
              "Video araması",
              "Transkript çekimi",
              "İçerik filtresi",
            ]}
            output="302 kayıt"
            onPreview={onPreview}
          />


          <CollectionCard
            icon="🏛"
            title="Resmî Kurum"
            subtitle="Kaynak odaklı script tabanlı toplama"
          />


          <CollectionCard
            icon="🎓"
            title="Akademik"
            subtitle="Çoklu akademik kaynak + otomatik / yarı otomatik toplama"
          />


        </div>


        <div className="collection-automation-note">

          <b>
            🖥 Arayüz ikonunu kullan:
          </b>

          <span>
            Ulusal Haber, Yerel Haber ve YouTube kartlarındaki
            arayüz ikonuna tıklandığında araştırmada kullanılan
            gerçek veri toplama ekranı, kullanılan parametreler
            ve üretilen çıktı sağ panelde gösterilir.
          </span>

        </div>


        <Flow
          items={[
            "Araştırmacı parametreleri belirler",
            "Otomasyon sorguları üretir",
            "Kaynak taranır",
            "Metin / transkript elde edilir",
            "Kalite ve tekilleştirme",
            "Kaynak alt korpusu",
          ]}
        />


        <IOGrid
          purpose="Çok kaynaklı web verisinin aynı araştırma mantığıyla tekrar toplanabilmesini ve veri üretiminin araştırmacı tarafından kontrol edilebilmesini sağlamak."
          input="Hedef tür, domain/anahtar kelime, sayfa veya kayıt sınırları ve çıktı klasörü gibi kullanıcı parametreleri."
          process="Form ve buton arayüzlerinden tetiklenen kaynak-özel toplama otomasyonları; akademik ve resmî kaynaklarda script/yarı otomatik toplama."
          output="Ulusal haber, yerel haber, YouTube, resmî ve akademik kaynak alt korpusları."
        />

      </>
    );
  }



  /* =======================================================
     NATIONAL
     ======================================================= */

  if (detail === "national") {

    return (
      <>

        <div className="detail-title-with-preview">

          <h2>
            Ulusal Haber Veri Toplama Otomasyonu
          </h2>

          <PreviewButton
            onClick={() =>
              onPreview({
                title: "Ulusal Haber",
                image: nationalCollectionUi,
                description:
                  "Ulusal haber kaynaklarında hedef tür ile ilişkili içeriklerin sistematik olarak toplanması.",
                features: [
                  "Tür adı ve anahtar kelimeler",
                  "Domain listesi",
                  "Sitemap taraması",
                  "URL keşfi",
                  "Full text çıkarımı",
                  "Batch işlemleri",
                ],
                output: "601 belge",
              })
            }
          />

        </div>


        <p className="research-lead">
          Ulusal medya kaynaklarındaki kahverengi kokarca
          haberlarını tek tek elle toplamak yerine, arama ve
          tam metin elde etme adımlarını sistematikleştiren
          bir haber toplama iş akışı kullanıldı.
        </p>


        <IOGrid
          purpose="Ulusal ölçekte yayımlanan olay, zarar, yayılım ve mücadele haberlarını araştırma korpusuna kazandırmak."
          input="Hedef tür ve ilişkili anahtar kelimeler, ulusal medya kaynakları ve haber URL'leri."
          process="Arama → aday URL → haber metni → uygunluk/kalite kontrolü → tekilleştirme."
          output="Ulusal haber alt korpusu."
        />

      </>
    );
  }



  /* =======================================================
     LOCAL
     ======================================================= */

  if (detail === "local") {

    return (
      <>

        <div className="detail-title-with-preview">

          <h2>
            Yerel Haber Veri Toplama Otomasyonu
          </h2>

          <PreviewButton
            onClick={() =>
              onPreview({
                title: "Yerel Haber",
                image: localCollectionUi,
                description:
                  "Yerel haber kaynaklarında il ve ilçe düzeyindeki gözlem, yayılım, zarar ve mücadele bilgilerinin sistematik olarak toplanması.",
                features: [
                  "Tür adı",
                  "Ek anahtar kelimeler",
                  "Yerel domain listesi",
                  "URL keşfi",
                  "Full text çıkarımı",
                  "Batch işlemleri",
                ],
                output: "583 belge",
              })
            }
          />

        </div>


        <p className="research-lead">
          Yerel haberler, il ve ilçe düzeyindeki gözlem ve
          mücadele ayrıntılarını yakalamak için ayrı bir veri
          kaynağı olarak ele alındı.
        </p>


        <IOGrid
          purpose="Ulusal haberlere yansımayan yerel mekânsal, zamansal ve kurumsal bilgileri yakalamak."
          input="Yerel haber kaynakları, hedef tür anahtar kelimeleri ve aday haber sayfaları."
          process="Kaynak odaklı arama → URL/metin elde etme → konu uygunluğu → yinelenen kayıt kontrolü."
          output="Yerel haber alt korpusu."
        />

      </>
    );
  }



  /* =======================================================
     YOUTUBE
     ======================================================= */

  if (detail === "youtube-collect") {

    return (
      <>

        <div className="detail-title-with-preview">

          <h2>
            YouTube Veri Toplama Otomasyonu
          </h2>

          <PreviewButton
            onClick={() =>
              onPreview({
                title: "YouTube",
                image: youtubeCollectionUi,
                description:
                  "YouTube videolarındaki saha anlatıları, uzman açıklamaları ve yerel gözlemlerin transkript üzerinden metinsel veriye dönüştürülmesi.",
                features: [
                  "İstilacı tür sorgusu",
                  "Sayfa sayısı",
                  "Minimum metin uzunluğu",
                  "Video araması",
                  "Transkript çekimi",
                  "İçerik filtresi",
                ],
                output: "302 kayıt",
              })
            }
          />

        </div>


        <p className="research-lead">
          Video içerikleri, haber metinlerinden farklı olarak
          konuşma dili ve transkript üzerinden bilgi taşır.
          Bu nedenle video ve transkript elde etme adımları
          ayrı bir iş akışı olarak tasarlandı.
        </p>


        <IOGrid
          purpose="Saha anlatıları, uzman açıklamaları ve yerel gözlemler gibi videoda kalan bilgileri metinsel analize dahil etmek."
          input="YouTube arama sonuçları, videolar ve erişilebilir transkriptler."
          process="Video keşfi → transkript elde etme → içerik uygunluğu → kayıt tekilleştirme."
          output="YouTube transkript alt korpusu."
        />

      </>
    );
  }



  /* =======================================================
     OFFICIAL
     ======================================================= */

  if (detail === "official") {

    return (
      <>

        <h2>
          Resmî Kurum Kaynakları
        </h2>

        <p className="research-lead">
          Tarım ve Orman Bakanlığı ile il/ilçe müdürlükleri
          gibi resmî kaynaklar; müdahale, eylem planı,
          mücadele ve kurumsal kararlar açısından ayrı
          bir kanıt katmanı olarak toplandı.
        </p>

        <IOGrid
          purpose="Kamu müdahalelerini ve resmî açıklamaları doğrudan birincil kurumsal kaynaklardan yakalamak."
          input="Resmî kurum siteleri ve hedef türle ilişkili içerik sayfaları."
          process="Kaynak odaklı arama → içerik elde etme → konu uygunluğu → kurumsal kayıt."
          output="Resmî kurum alt korpusu."
        />

      </>
    );
  }



  /* =======================================================
     ACADEMIC
     ======================================================= */

  if (detail === "academic") {

    return (
      <>

        <h2>
          Akademik Veri Toplama
        </h2>

        <p className="research-lead">
          Akademik kaynaklar tek bir servisle sınırlı tutulmadı;
          DergiPark, OpenAlex, TR Dizin ve Web of Science gibi
          farklı yollarla metadata ve erişilebilen tam metinler
          araştırma korpusuna dahil edildi.
        </p>


        <div className="pill-row">
          <span>DergiPark</span>
          <span>OpenAlex</span>
          <span>TR Dizin</span>
          <span>Web of Science</span>
          <span>Tam metin / PDF</span>
        </div>


        <IOGrid
          purpose="Haber ve resmî metinlerdeki anlatıları bilimsel literatürle tamamlamak."
          input="Akademik arama sorguları, bibliyografik kayıtlar ve erişilebilir makale metinleri."
          process="Kaynak araması → metadata eşleme → erişilebilir tam metin/PDF → akademik korpus."
          output="Akademik alt korpus."
        />

      </>
    );
  }



  /* =======================================================
     CORPUS
     ======================================================= */

  if (detail === "corpus") {

    return (
      <>

        <h2>
          2036 Belgelik Analiz Korpusu
        </h2>

        <p className="research-lead">
          Kaynak-özel toplama süreçleri sonunda farklı yapıdaki
          belgeler ortak kimlik ve kaynak alanlarıyla tek analiz
          korpusunda birleştirildi.
        </p>


        <div className="big-number-card">
          <b>2036</b>
          <span>çok kaynaklı belge</span>
        </div>


        <Flow
          items={[
            "Ulusal",
            "Yerel",
            "YouTube",
            "Akademik",
            "Resmî",
            "Ortak analiz korpusu",
          ]}
        />


        <IOGrid
          purpose="LLM tabanlı bilgi çıkarımı için tek ve izlenebilir bir belge tabanı oluşturmak."
          input="Beş kaynak türünden elde edilen belge alt korpusları."
          process="Kimliklendirme, kaynak türü koruma, birleştirme ve son veri kontrolü."
          output="LLM çıkarım hattının girdisi olan ana korpus."
        />

      </>
    );
  }



  /* =======================================================
     LLM OVERVIEW
     ======================================================= */

  if (false && detail === "llm-overview") {

    return (
      <>

        <h2>
          LLM ile Bilgi Çıkarımı
        </h2>

        <p className="research-lead">
          Bu aşamanın temel amacı metinleri özetlemek değil;
          serbest metinlerde saklı kalan mekânsal, zamansal,
          kurumsal, müdahale ve iklim bilgilerini yapılandırılmış
          alanlara dönüştürmektir.
        </p>


        <Flow
          items={[
            "2036 belge",
            "Kaynak bazlı hazırlık",
            "Hedef tür ön filtresi",
            "Prompt + GPT-4.1-mini",
            "JSON Schema",
            "Güvenlik / kalite kontrolleri",
            "Yapısal olay verisi",
          ]}
        />


        <IOGrid
          purpose="Metinlerde saklı bilgileri analiz edilebilir, sorgulanabilir ve haritalanabilir verilere dönüştürmek."
          input="Çok kaynaklı belge korpusu."
          process="Kurallı prompt tasarımı + yapılandırılmış LLM çıkarımı + post-processing."
          output="Mekân, zaman, ürün, müdahale, kurum, iklim ve kanıt alanları içeren yapısal kayıtlar."
        />

      </>
    );
  }



  /* =======================================================
     YOUTUBE PREPROCESS
     ======================================================= */

  if (detail === "youtube-preprocess") {

    return (
      <>

        <h2>
          Kaynak Bazlı Ön İşleme
        </h2>

        <p className="research-lead">
          Genel haber, akademik ve resmî metinler özgün
          içerikleri korunarak LLM'e aktarılırken,
          yalnızca YouTube transkriptlerinde konuşma dili
          ve transkript kaynaklı hatalara yönelik hafif
          normalizasyon uygulandı.
        </p>


        <div className="transform-grid">

          <div>
            <small>HAM TRANSKRİPT</small>
            <code>iki bin yirmi üç</code>
          </div>

          <b>→</b>

          <div>
            <small>LLM İÇİN</small>
            <code>2023</code>
          </div>


          <div>
            <small>HAM TRANSKRİPT</small>
            <code>kahve rengi kokarca</code>
          </div>

          <b>→</b>

          <div>
            <small>LLM İÇİN</small>
            <code>kahverengi kokarca</code>
          </div>


          <div>
            <small>HAM TRANSKRİPT</small>
            <code>halyomorfa halis</code>
          </div>

          <b>→</b>

          <div>
            <small>LLM İÇİN</small>
            <code>halyomorpha halys</code>
          </div>

        </div>


        <IOGrid
          purpose="YouTube konuşma/transkript hatalarının yıl ve tür çıkarımını etkilemesini azaltmak."
          input="YouTube transkript metni."
          process="Türkçe karakter standardizasyonu, sözel yıl kalıpları ve bilinen tür adı transkript hatalarının düzeltilmesi."
          output="Anlamı korunmuş, LLM analizine hazırlanmış YouTube metni."
        />

      </>
    );
  }



  /* =======================================================
     PREFILTER
     ======================================================= */

  if (detail === "prefilter") {

    return (
      <>

        <h2>
          Hedef Tür Ön Filtresi
        </h2>

        <p className="research-lead">
          Ayrıntılı LLM çıkarımından önce belgenin hedef
          türle ilişkili olup olmadığı çekirdek terimler
          üzerinden kontrol edildi.
        </p>


        <div className="yes-no-grid">

          <div className="yes">

            <b>ANALİZE AL</b>

            <span>kahverengi kokarca</span>
            <span>Halyomorpha halys</span>
            <span>H. halys</span>
            <span>Brown marmorated stink bug</span>
            <span>BMSB</span>

          </div>


          <div className="no">

            <b>HEDEF TÜR DEĞİL</b>

            <span>yeşil kokarca</span>
            <span>Palomena prasina</span>
            <span>kuzalak akarı</span>

          </div>

        </div>


        <IOGrid
          purpose="LLM çıkarımını hedef türle doğrudan ilişkili metinler üzerinde yoğunlaştırmak."
          input="Analiz korpusundaki belge metni."
          process="Çekirdek hedef tür terimleri ve diğer-zararlı kontrolü."
          output="GPT çıkarımına gönderilecek uygun belgeler veya atlama durumu."
        />

      </>
    );
  }



  /* =======================================================
     PROMPT
     ======================================================= */

 if (detail === "prompt") {
  return (
    <>
      <h2>
        Prompt Tasarımı
      </h2>

      <p className="research-lead">
        Modelin serbest yanıt üretmesi yerine araştırma sorularına uygun,
        izlenebilir ve yapılandırılmış bilgi çıkarması için System Prompt,
        User Prompt ve JSON Schema birlikte tasarlandı.
      </p>

      <div className="prompt-detail-grid">

        {/* SYSTEM PROMPT */}
        <article className="prompt-detail-card">

          <div className="prompt-detail-card-head">

            <div>
              <span className="prompt-detail-icon">
                🧠
              </span>

              <div>
                <b>
                  System Prompt
                </b>

                <small>
                  Model rolü + çıkarım ve güvenlik kuralları
                </small>
              </div>
            </div>

            <PreviewButton
              label="Prompt"
              icon="▣"
              onClick={() =>
                onPreview({
                  title: "System Prompt",
                  kicker: "LLM BİLGİ ÇIKARIM MİMARİSİ",
                  description:
                    "System Prompt, modelin araştırmadaki rolünü ve bilgi çıkarım sınırlarını tanımlar.",
                  contentTitle: "System Prompt — yapı",
                  content: `ROL
Çok kaynaklı çevresel metinlerden yapılandırılmış araştırma bilgisi çıkar.

TEMEL KURAL
Yalnızca verilen belge metninde açıkça desteklenen bilgileri çıkar.
Metinde bulunmayan bilgiyi tahmin etme veya tamamlama.

MEKÂN
Belgede açıkça belirtilen il, ilçe ve yer adlarını çıkar.
Belirsiz veya genel bölge ifadelerini kesin idari konum olarak üretme.

ZAMAN
Belge yayın tarihi ile olay yılını birbirinden ayır.
Olay yılı açık değilse tahmin üretme.

MÜDAHALE
Kurum, devlet müdahalesi ve mücadele türlerini yalnızca açık kanıt varsa çıkar.

İKLİM
Normal mevsimsel davranışı otomatik olarak iklim değişikliği etkisi olarak sınıflandırma.
İklim ilişkisini yalnızca metinde açık bir bağlantı kurulmuşsa kaydet.

KANIT
Önemli çıkarımları destekleyen evidence_sentence alanını doldur.

BELİRSİZLİK
Bilgi açık değilse null, boş liste veya uygun belirsizlik değeri kullan.

ÇIKTI
Yalnızca tanımlanan JSON Schema ile uyumlu yapılandırılmış çıktı üret.`,
                  contentLanguage: "text",
                })
              }
            />

          </div>

          <p>
            Modelin rolünü, yalnızca metinde desteklenen bilgiyi çıkarma
            ilkesini ve mekân-zaman-iklim güvenlik kurallarını tanımlar.
          </p>

        </article>


        {/* USER PROMPT */}
        <article className="prompt-detail-card">

          <div className="prompt-detail-card-head">

            <div>
              <span className="prompt-detail-icon">
                💬
              </span>

              <div>
                <b>
                  User Prompt
                </b>

                <small>
                  Belge metadata + analiz edilecek metin
                </small>
              </div>
            </div>

            <PreviewButton
              label="Prompt"
              icon="▣"
              onClick={() =>
                onPreview({
                  title: "User Prompt",
                  kicker: "BELGE BAZLI ANALİZ GİRDİSİ",
                  description:
                    "User Prompt her belge için kimlik, kaynak türü ve analiz edilecek metni modele aktarır.",
                  contentTitle: "User Prompt — örnek yapı",
                  content: `BELGE BİLGİLERİ

global_id: {{global_id}}
doc_id: {{doc_id}}
source_type: {{source_type}}
title: {{title}}
published_date: {{published_date}}
url: {{url}}

ANALİZ EDİLECEK METİN

{{document_text}}

GÖREV

Yukarıdaki metni System Prompt kurallarına göre analiz et.

Metinde açıkça desteklenen:
- mekânsal bilgileri,
- olay / gözlem zamanını,
- ürün / konukçu bilgisini,
- olay türünü,
- kurum ve müdahale bilgisini,
- iklim sinyalini,
- iklim yorumunu,
- eğilim yönünü,
- destekleyici kanıt cümlesini

çıkar.

Metinde bulunmayan bilgileri üretme.

Kaynak türü YouTube ise transkript kaynaklı yazım ve sözel yıl biçimlerini bağlam içinde değerlendir.

Yanıtı yalnızca tanımlanan JSON Schema ile uyumlu biçimde üret.`,
                  contentLanguage: "text",
                })
              }
            />

          </div>

          <p>
            Belgenin kimlik bilgileri, kaynak türü ve özgün metni
            dinamik biçimde modele gönderilir.
          </p>

        </article>


        {/* JSON SCHEMA */}
        <article className="prompt-detail-card">

          <div className="prompt-detail-card-head">

            <div>
              <span className="prompt-detail-icon">
                📋
              </span>

              <div>
                <b>
                  JSON Schema
                </b>

                <small>
                  Yapılandırılmış çıktı sözleşmesi
                </small>
              </div>
            </div>

            <PreviewButton
              label="Schema"
              icon="{ }"
              onClick={() =>
                onPreview({
                  title: "JSON Schema",
                  kicker: "STRUCTURED OUTPUT",
                  description:
                    "JSON Schema modelin serbest metin yerine araştırmada kullanılacak alanları aynı veri yapısıyla üretmesini sağlar.",
                  contentTitle: "Temel çıkarım alanları",
                  content: `{
  "global_id": "string",
  "source_type": "string",
  "summary_tr": "string | null",

  "location_mentions": [],
  "provinces_raw": [],
  "districts_raw": [],

  "spread_observation_year": "number | null",
  "intervention_year": "number | null",

  "crops_raw": [],
  "event_type_raw": [],

  "government_action_raw": [],
  "institution_raw": [],

  "climate_signal_raw": [],
  "climate_factor_type_raw": [],
  "climate_comment": "string | null",

  "trend_direction": "string | null",

  "extraction_confidence": "string | null",
  "evidence_sentence": "string | null"
}`,
                  contentLanguage: "json",
                })
              }
            />

          </div>

          <p>
            Alan tipleri, null değerleri ve beklenen yapısal çıktı
            schema tarafından sınırlandırılır.
          </p>

        </article>


        {/* TASARIM İLKELERİ */}
        <article className="prompt-detail-card">

          <div className="prompt-detail-card-head">

            <div>
              <span className="prompt-detail-icon">
                🛡
              </span>

              <div>
                <b>
                  Tasarım İlkeleri
                </b>

                <small>
                  Yanlış pozitifleri azaltan kurallar
                </small>
              </div>
            </div>

          </div>

          <ul className="prompt-rule-list">
            <li>Tahmin üretme.</li>
            <li>Belirsizlikte null / boş alan kullan.</li>
            <li>Haber tarihini otomatik olay yılı yapma.</li>
            <li>Belirsiz bölgeyi kesin idari konuma dönüştürme.</li>
            <li>Normal mevsimsel davranışı otomatik iklim etkisi sayma.</li>
          </ul>

        </article>

      </div>


      <div className="field-cloud">

        {extractionFields.map((field) => (
          <code key={field}>
            {field}
          </code>
        ))}

      </div>
    </>
  );
}

  /* =======================================================
     EXTRACTION
     ======================================================= */

  if (detail === "extraction") {

    return (
      <>

        <h2>
          GPT-4.1-mini ile Yapılandırılmış Bilgi Çıkarımı
        </h2>

        <p className="research-lead">
          Model, belgeyi yalnızca özetlemez; aynı metinden
          birden fazla araştırma boyutunu aynı yapılandırılmış
          kayıt içinde çıkarır.
        </p>


        <div className="document-to-json">

          <div>

            <small>
              ÖRNEK METİN
            </small>

            <p>
              “2024 yılında Ordu’nun Fatsa ilçesinde
              kahverengi kokarca nedeniyle fındıkta zarar
              bildirildi. İl Tarım ve Orman Müdürlüğü
              biyolojik mücadele kapsamında samuray arısı
              salımı yapılacağını açıkladı. Uzmanlar ılıman
              kış koşullarının popülasyonu artırdığını belirtti.”
            </p>

          </div>


          <b>
            →
          </b>


          <pre>
{`{
  "provinces_raw": ["Ordu"],
  "districts_raw": ["Fatsa"],
  "spread_observation_year": 2024,
  "crops_raw": ["fındık"],
  "event_type_raw": ["zarar", "mücadele"],
  "institution_raw": ["İl Tarım ve Orman Müdürlüğü"],
  "climate_signal_raw": ["ılıman kış"],
  "trend_direction": "artış"
}`}
          </pre>

        </div>


        <IOGrid
          purpose="Tek bir metindeki saklı araştırma bilgisini çok boyutlu yapısal kayda dönüştürmek."
          input="Filtrelenmiş belge + system prompt + user prompt."
          process="GPT-4.1-mini ile strict JSON Schema tabanlı extraction."
          output="Yapılandırılmış çıkarım + evidence_sentence + extraction_confidence."
        />

      </>
    );
  }



  /* =======================================================
     POSTPROCESS
     ======================================================= */

  if (detail === "postprocess") {

    return (
      <>

        <h2>
          Kalite ve Güvenlik Kontrolleri
        </h2>

        <p className="research-lead">
          LLM çıktısı doğrudan nihai veri kabul edilmedi.
          Çıkarımdan sonra konum ve iklim alanlarında
          kural tabanlı güvenlik kontrolleri uygulandı.
        </p>


        <div className="safety-grid">

          <div>
            <b>📍 Lokasyon</b>
            <span>
              Türkiye illeri doğrulanır; genel bölgeler,
              belirsiz yerler ve Türkiye dışı lokasyonlar ayrılır.
            </span>
          </div>


          <div>
            <b>◷ Zaman</b>
            <span>
              Haber tarihi, olay yılı ve müdahale yılı
              birbirinden ayrılır.
            </span>
          </div>


          <div>
            <b>🌦 Mevsim / iklim</b>
            <span>
              “15 derece”, “ilkbahar”, “kışlak” gibi
              normal mevsimsel davranışlar otomatik
              iklim etkisi yapılmaz.
            </span>
          </div>


          <div>
            <b>🛡 İlgisiz kayıt</b>
            <span>
              İlgisiz sınıflandırılan belgelerde mekân,
              zaman, ürün, müdahale ve iklim alanları temizlenir.
            </span>
          </div>

        </div>


        <IOGrid
          purpose="Model çıktısındaki mekânsal, zamansal ve iklimsel yanlış pozitifleri azaltmak."
          input="GPT tarafından üretilen JSON."
          process="Post-processing, idari konum doğrulama, mevsim/iklim ayrımı ve safety locks."
          output="Analiz için temizlenmiş ve daha tutarlı yapısal kayıt."
        />

      </>
    );
  }



  /* =======================================================
     GENERAL AUTOMATION
     ======================================================= */

 if (false && detail === "automation") {
  return (
    <>
      <div className="detail-title-with-preview">
        <div>
          <h2>Genellenebilir GPT Analiz Otomasyonu</h2>

          <p className="research-lead">
            İlk kahverengi kokarca bilgi çıkarım hattı daha sonra farklı türler
            ve veri kümeleri üzerinde yeniden kullanılabilecek parametrik bir
            analiz aracına dönüştürüldü.
          </p>
        </div>

        <PreviewButton
          label="Otomasyon Arayüzlerini Gör"
          icon="▣"
          onClick={() =>
            onPreview({
              title: "Genellenebilir GPT Analiz Otomasyonu",
              kicker: "ARAŞTIRMA OTOMASYONU",

              description:
                "Geliştirilen arayüz, araştırmacının proje ve hedef tür bilgilerini tanımlamasına, CSV veri setini yüklemesine ve veri kolonlarını analiz alanlarıyla eşleştirmesine olanak sağlar.",

              images: [
                {
                  src: gptAutomationSetup,
                  title: "1. Proje ve hedef tür ayarları",
                  caption:
                    "Proje adı, hedef tür, bilimsel ad, eş adlar, çalışma alanı, çalışma dönemi ve araştırma amacı kullanıcı tarafından tanımlanır.",
                },
                {
                  src: gptAutomationMapping,
                  title: "2. Veri dosyası yükleme ve kolon eşleme",
                  caption:
                    "CSV veri dosyası sisteme yüklenir ve metin, başlık, tarih ve URL kolonları arayüz üzerinden eşleştirilir.",
                },
              ],

              features: [
                "Proje tanımlama",
                "Hedef tür tanımlama",
                "Bilimsel ad ve eş adlar",
                "Çalışma alanı ve dönemi",
                "CSV yükleme",
                "Kolon eşleme",
                "Test modu",
                "Tam analiz",
              ],

              output:
                "Yeniden kullanılabilir GPT tabanlı bilgi çıkarım otomasyonu",
            })
          }
        />
      </div>

      <Flow
        items={[
          "Proje / hedef tür ayarları",
          "CSV yükleme",
          "Kolon eşleme",
          "Dinamik prompt",
          "LLM Bilgi Çıkarımı",
          "Yapısal çıktı",
          "Kalite raporu / analiz tablosu",
        ]}
      />

         <div className="automation-capability-grid">

        <div>
          <span>⚙</span>

          <b>
            Parametrik
          </b>

          <small>
            Proje ve hedef tür kod değiştirmeden tanımlanabilir.
          </small>
        </div>


        <div>
          <span>📄</span>

          <b>
            CSV tabanlı
          </b>

          <small>
            Farklı veri kümeleri sisteme yüklenebilir.
          </small>
        </div>


        <div>
          <span>⇄</span>

          <b>
            Kolon eşleme
          </b>

          <small>
            Veri setindeki metin, başlık, tarih ve URL alanları
            arayüzden eşleştirilir.
          </small>
        </div>


        <div>
          <span>🤖</span>

          <b>
            Dinamik LLM analizi
          </b>

          <small>
            Prompt ve çıkarım hattı tanımlanan proje bağlamına göre çalışır.
          </small>
        </div>

      </div>


      <IOGrid
        purpose="Geliştirilen bilgi çıkarım yöntemini yeniden kullanılabilir bir araştırma aracına dönüştürmek."
        input="Kullanıcının belirlediği proje/tür bilgileri ve analiz edilecek CSV."
        process="Parametrik yapılandırma, kolon eşleme, dinamik prompt üretimi ve otomatik LLM analizi."
        output="Yeni veri setleri için yeniden üretilebilir yapılandırılmış extraction iş akışı."
      />

    </>
  );
}

  /* =======================================================
     CLIMATE OVERVIEW
     ======================================================= */

  if (detail === "climate-overview") {

    return (
      <>

        <h2>
          İklim ve İklim Değişikliği Analizi
        </h2>

        <p className="research-lead">
          Bu bölümde iki kavram birlikte araştırıldı:
          metinlerdeki doğrudan iklim bilgileri ve
          bu bilgilerin iklim değişikliğiyle bilimsel
          olarak ilişkilendirilip ilişkilendirilemeyeceği.
        </p>


        <Flow
          items={[
            "Metinsel iklim sinyalleri",
            "ClimateGPT değerlendirmesi",
            "Bilimsel iklim bağlamı",
            "ERA5 anomalileri",
            "Tutarlılık analizi",
          ]}
        />


        <IOGrid
          purpose="Kahverengi kokarca ile iklim/iklim değişikliği arasındaki olası ilişkinin çok katmanlı biçimde incelenmesi."
          input="LLM ile çıkarılan climate_signal, climate_comment, evidence ve il-yıl bilgileri."
          process="ClimateGPT model değerlendirmesi + araştırma kullanımı + ERA5 gözlemsel karşılaştırma."
          output="İklim anlatıları, bilimsel bağlam, gözlemsel anomaliler ve tutarlılık sınıfları."
        />

      </>
    );
  }



  /* =======================================================
     CLIMATEGPT EVALUATION
     ======================================================= */

  if (detail === "climategpt-eval") {

    return (
      <>

        <h2>
          ClimateGPT Model Değerlendirmesi
        </h2>

        <p className="research-lead">
          ClimateGPT yalnızca araştırma verisine uygulanmadı;
          önce modelin kahverengi kokarca–iklim–iklim değişikliği
          ilişkisini bilimsel bağlamda açıklama kapasitesi ayrıca incelendi.
        </p>


        <div className="dual-eval">

          <div>
            <b>☁ RunPod</b>
            <span>
              ClimateGPT-7B modeli çalıştırıldı ve
              bilimsel iklim soruları test edildi.
            </span>
          </div>


          <div>
            <b>🌐 ClimateGPT Web</b>
            <span>
              Aynı problem alanında web arayüzünün
              yanıt, açıklama ve referans davranışı incelendi.
            </span>
          </div>

        </div>


        <div className="research-question-card">
          “ClimateGPT, kahverengi kokarca ile iklim ve
          iklim değişikliği arasındaki ilişki hakkında
          kullanılabilir bilimsel bilgi sunabiliyor mu?”
        </div>


        <IOGrid
          purpose="ClimateGPT'nin araştırma problemine uygunluğunu doğrudan soru-cevap üzerinden değerlendirmek."
          input="Kahverengi kokarca, sıcaklık, ılıman kış, yayılım ve iklim değişikliği ilişkisine yönelik bilimsel sorular."
          process="RunPod ClimateGPT-7B ve ClimateGPT web arayüzündeki yanıtların incelenmesi."
          output="Modelin bilimsel iklim bilgisini açıklama ve ilişki kurma kapasitesine dair değerlendirme."
        />

      </>
    );
  }



  /* =======================================================
     CLIMATEGPT RESEARCH
     ======================================================= */

  if (detail === "climategpt-research") {

    return (
      <>

        <h2>
          ClimateGPT'nin Araştırma İçinde Kullanılması
        </h2>

        <p className="research-lead">
          Model değerlendirmesinden ayrı olarak,
          metinlerden çıkarılan iklim anlatılarının
          bilimsel iklim bağlamında değerlendirilmesi için
          ClimateGPT/RAG tabanlı kanıt paketleri kullanıldı.
        </p>


        <Flow
          items={[
            "Event / climate narrative",
            "Evidence package",
            "ClimateGPT",
            "Province–Year card",
            "Bilimsel iklim bağlamı",
          ]}
        />


        <IOGrid
          purpose="Metindeki iklim anlatısının yalnızca haber söylemi olarak kalmayıp bilimsel iklim bilgisi çerçevesinde yorumlanması."
          input="İklim sinyali, kanıt cümlesi, olay/iklim yılı ve il bağlamı."
          process="Evidence package ve province–year RAG kartları üzerinden ClimateGPT değerlendirmesi."
          output="Kahverengi kokarca ile iklim/iklim değişikliği ilişkisinin bilimsel bağlamda yorumlanmış anlatısı."
        />

      </>
    );
  }



  /* =======================================================
     ERA5
     ======================================================= */

  if (detail === "era5") {

    return (
      <>

        <h2>
          ERA5 ile Gözlemsel İklim Karşılaştırması
        </h2>

        <p className="research-lead">
          ERA5, ECMWF tarafından üretilen atmosferik yeniden
          analiz (reanalysis) veri setidir. Çalışmada metinsel
          iddialardan bağımsız gözlemsel iklim referansı
          sağlamak için kullanıldı.
        </p>


        <div className="climate-vars">

          <div>
            🌡
            <b>Yıllık sıcaklık anomalisi</b>
          </div>

          <div>
            ❄
            <b>Mevsimsel sıcaklık anomalileri</b>
          </div>

          <div>
            💧
            <b>Bağıl nem anomalisi</b>
          </div>

          <div>
            🌧
            <b>Yağış anomalisi</b>
          </div>

        </div>


        <IOGrid
          purpose="Metinlerde ifade edilen iklim koşullarını aynı il ve yıl için gözlenen iklim sinyalleriyle karşılaştırmak."
          input="İl, iklim yılı ve iklim faktörü."
          process="ERA5 sıcaklık, yağış ve bağıl nem anomali değerlerinin eşleştirilmesi."
          output="Metinsel iklim anlatısına eşlik eden bağımsız gözlemsel iklim göstergeleri."
        />

      </>
    );
  }



  /* =======================================================
     CONSISTENCY
     ======================================================= */

  if (detail === "consistency") {

    return (
      <>

        <h2>
          Tutarlılık Analizi
        </h2>

        <p className="research-lead">
          Amaç metnin doğru/yanlış olduğunu ilan etmek değil;
          metindeki iklim anlatısının aynı yer ve zamandaki
          gözlemsel iklim sinyaliyle ne ölçüde uyumlu
          olduğunu değerlendirmektir.
        </p>


        <div className="consistency-grid">

          <div className="good">
            <b>● Tutarlı</b>
            <span>
              Metinsel anlatı gözlenen iklim sinyaliyle destekleniyor.
            </span>
          </div>


          <div className="partial">
            <b>● Kısmen tutarlı</b>
            <span>
              Anlatının yalnızca bir bölümü veya
              dönemsel bileşeni destekleniyor.
            </span>
          </div>


          <div className="bad">
            <b>● Tutarlı değil</b>
            <span>
              Gözlenen iklim sinyali anlatıyı desteklemiyor.
            </span>
          </div>


          <div className="neutral">
            <b>● Doğrudan test edilemez</b>
            <span>
              Anlatı ERA5 ile sınanabilecek belirgin
              gözlemsel iddia içermiyor.
            </span>
          </div>

        </div>


        <IOGrid
          purpose="Metinsel iklim bilgisi ile gözlemsel iklim verisi arasındaki uyumu şeffaf biçimde göstermek."
          input="Climate narrative + ERA5 anomalileri."
          process="Faktör, yön, yıl ve gözlenen anomali bağlamında karşılaştırma."
          output="Tutarlılık sınıfı ve gerekçe."
        />

      </>
    );
  }



  /* =======================================================
     GEOVISUAL OVERVIEW
     ======================================================= */

  if (detail === "geo-overview") {

    return (
      <>

        <h2>
          Bağlantılı Jeogörsel Analiz
        </h2>

        <p className="research-lead">
          Son aşamada yapısal olay, belge, iklim ve müdahale
          bilgileri tek bir statik haritada gösterilmek yerine,
          birbirini güncelleyen bağlantılı görünümler içinde analiz edilir.
        </p>


        <div className="geo-module-grid">

          <div>
            🗺
            <b>Harita</b>
            <span>Mekânsal dağılım</span>
          </div>

          <div>
            ◷
            <b>Zaman</b>
            <span>Zamansal değişim</span>
          </div>

          <div>
            📄
            <b>Belgeler</b>
            <span>Kanıta geri dönüş</span>
          </div>

          <div>
            🌦
            <b>İklim</b>
            <span>ClimateGPT + ERA5</span>
          </div>

          <div>
            🏛
            <b>Müdahaleler</b>
            <span>Kurum ve eylem</span>
          </div>

          <div>
            🧊
            <b>STC</b>
            <span>3B mekân-zaman</span>
          </div>

          <div>
            ?
            <b>Soru-Cevap</b>
            <span>Veriye dayalı sorgulama</span>
          </div>

        </div>


        <IOGrid
          purpose="Mekânsal, zamansal, metinsel, kurumsal ve iklimsel ilişkileri birlikte keşfetmek."
          input="Yapısal olay verisi ve ilişkili analiz katmanları."
          process="Ortak seçim ve filtre durumunu paylaşan coordinated / linked views."
          output="Etkileşimli araştırma ve çok boyutlu karşılaştırma ortamı."
        />

      </>
    );
  }



  /* =======================================================
     GEOVISUAL DETAILS
     ======================================================= */

  const geoDetails:
    Record<
      string,
      [string, string, string, string]
    > = {

      map: [
        "Harita",
        "Olayların mekânsal dağılımını ve seçili zaman filtresindeki yoğunluğu incelemek.",
        "İl ve olay kayıtları.",
        "İl seçimi diğer bağlı görünümleri günceller.",
      ],

      timeline: [
        "Zaman Görünümü",
        "Olay, yayın, müdahale ve iklim zamanlarını aynı analiz bağlamında incelemek.",
        "Yıllara göre özetlenmiş kayıtlar.",
        "Yıl / aralık seçimi diğer görünümleri filtreler.",
      ],

      documents: [
        "Belgeler",
        "Yapısal veriden orijinal kanıt metnine geri dönebilmek.",
        "Belge–olay bağlantıları.",
        "Kaynak, özet, kanıt cümlesi ve URL üzerinden izlenebilirlik.",
      ],

      "climate-view": [
        "İklim",
        "Metinsel iklim anlatıları, ClimateGPT değerlendirmeleri ve ERA5 göstergelerini birlikte incelemek.",
        "İklim evidence kayıtları.",
        "İklim faktörü, anomali ve tutarlılık bağlamında analitik görünüm.",
      ],

      interventions: [
        "Müdahaleler",
        "Kurumların mücadele türlerini ve zamanlamasını olaylarla ilişkilendirmek.",
        "Müdahale aktörü, türü, yıl ve konum.",
        "Kurumsal müdahale bağlamı.",
      ],

      stc: [
        "Space-Time Cube (STC)",
        "Olayların mekânsal ve zamansal yoğunluğunu üç boyutlu olarak incelemek.",
        "İl × yıl olay yoğunluğu.",
        "Mekân-zaman örüntülerinin 3B görünümü.",
      ],

      questions: [
        "Soru-Cevap",
        "Kullanıcının yapılandırılmış araştırma verisi üzerinde doğal dille sorgulama yapmasını desteklemek.",
        "Bağlı veri / veri tabanı sorguları.",
        "Veriye dayalı yanıt ve ilgili kayıtlar.",
      ],
    };


  const item = geoDetails[detail];


  if (item) {

    return (
      <>

        <h2>
          {item[0]}
        </h2>

        <p className="research-lead">
          {item[1]}
        </p>

        <IOGrid
          purpose={item[1]}
          input={item[2]}
          process="Bağlantılı seçim, filtreleme ve etkileşim."
          output={item[3]}
        />

      </>
    );
  }


  return null;
}


/* =========================================================
   MAIN VIEW
   ========================================================= */

export default function MethodologyView() {

  const [section, setSection] =
    useState<SectionKey>("data");

  const [detail, setDetail] =
    useState<DetailKey>("data-overview");

  const [preview, setPreview] =
    useState<PreviewData | null>(null);


  const activeSection = useMemo(
    () =>
      SECTIONS.find(
        (item) => item.key === section
      )!,
    [section]
  );


  function selectSection(
    key: SectionKey
  ) {

    setSection(key);

    const first =
      SECTIONS.find(
        (item) => item.key === key
      )!.items[0].key;

    setDetail(first);
  }


  return (
    <section className="research-process">


      {/* HEADER */}

      <header className="research-process-header">

        <h1>
          Araştırma Süreci
        </h1>

      </header>



      {/* MAIN SECTION TABS */}

      <div className="research-section-tabs">

        {SECTIONS.map((item) => (

          <button
            key={item.key}
            className={
              section === item.key
                ? "active"
                : ""
            }
            onClick={() =>
              selectSection(item.key)
            }
          >

            <span>
              {item.icon}
            </span>

            <b>
              {item.label}
            </b>

          </button>

        ))}

      </div>



      {/* MAIN LAYOUT */}

      <div className="research-process-layout">


        {/* LEFT NAVIGATION */}

        <aside className="research-step-nav">

          {activeSection.items.map((item) => (

            <button
              key={item.key}
              className={
                detail === item.key
                  ? "active"
                  : ""
              }
              onClick={() =>
                setDetail(item.key)
              }
            >

              <span>
                {item.icon}
              </span>


              <div>

                <b>
                  {item.label}
                </b>

                {item.subtitle && (
                  <small>
                    {item.subtitle}
                  </small>
                )}

              </div>

            </button>

          ))}

        </aside>



        {/* CONTENT */}

        <main className="research-process-content">

          <Detail
            detail={detail}
            onPreview={setPreview}
          />

        </main>


      </div>



      {/* ===================================================
          PREVIEW DRAWER
          =================================================== */}

      {preview && (
        <>


          {/* BACKDROP */}

          <div
            className="method-drawer-backdrop"
            onClick={() =>
              setPreview(null)
            }
          />


          {/* DRAWER */}

          <aside className="method-preview-drawer">


            {/* DRAWER HEADER */}

            <div className="method-preview-drawer-head">

              <div>

                <span className="method-preview-kicker">
                  {preview.kicker || "ARAŞTIRMA ARAYÜZÜ"}
                </span>

                <h3>
                  {preview.title}
                </h3>

              </div>


              <button
                type="button"
                className="method-preview-close"
                onClick={() =>
                  setPreview(null)
                }
                aria-label="Önizlemeyi kapat"
              >
                ×
              </button>

            </div>



           {/* =====================================================
    SINGLE IMAGE
    Ulusal / Yerel / YouTube gibi eski tek görsel yapısı
    ===================================================== */}

{preview.image && (
  <div className="method-preview-image">
    <img
      src={preview.image}
      alt={`${preview.title} kullanıcı arayüzü`}
    />
  </div>
)}


{/* =====================================================
    MULTIPLE IMAGES
    GPT otomasyonu gibi çoklu ekran görüntüsü
    ===================================================== */}

{preview.images && preview.images.length > 0 && (
  <div className="method-preview-gallery">

    {preview.images.map((image, index) => (
      <section
        className="method-preview-gallery-item"
        key={`${image.src}-${index}`}
      >

        {image.title && (
          <h4>
            {image.title}
          </h4>
        )}

        <div className="method-preview-image">
          <img
            src={image.src}
            alt={image.title || preview.title}
          />
        </div>

        {image.caption && (
          <p className="method-preview-caption">
            {image.caption}
          </p>
        )}

      </section>
    ))}

  </div>
)}


{/* =====================================================
    DESCRIPTION
    ===================================================== */}

{preview.description && (
  <p className="method-preview-description">
    {preview.description}
  </p>
)}


{/* =====================================================
    PROMPT / JSON SCHEMA / TEXT CONTENT
    ===================================================== */}

{preview.content && (
  <div className="method-preview-code-section">

    {preview.contentTitle && (
      <span>
        {preview.contentTitle}
      </span>
    )}

    <pre
      className={
        preview.contentLanguage === "json"
          ? "method-preview-code json"
          : "method-preview-code"
      }
    >
      {preview.content}
    </pre>

  </div>
)}


{/* =====================================================
    FEATURES
    ===================================================== */}

{preview.features && preview.features.length > 0 && (
  <div className="method-preview-features">

    <span>
      {preview.images
        ? "OTOMASYON ÖZELLİKLERİ"
        : "ÖZELLİKLER"}
    </span>

    <ul>

      {preview.features.map((feature) => (
        <li key={feature}>
          <i>✓</i>
          {feature}
        </li>
      ))}

    </ul>

  </div>
)}


{/* =====================================================
    OUTPUT
    ===================================================== */}

{preview.output && (
  <div className="method-preview-output">

    <span>
      BU AŞAMANIN ÇIKTISI
    </span>

    <strong>
      {preview.output}
    </strong>

  </div>
)}
          </aside>

        </>
      )}

    </section>
  );
}


