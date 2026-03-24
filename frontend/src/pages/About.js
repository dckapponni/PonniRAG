import React from 'react';

// API base URL — same pattern as api.js
const API_BASE = process.env.REACT_APP_API_URL || '';

const About = ({ language }) => {
  const ta = language === 'ta';

  return (
    <div className="about-container">
      <div className="about-header">
        <h2 className="about-heading">{ta ? 'பொன்னி களஞ்சியம்' : 'Ponni Archive'}</h2>
        <p className="about-subtitle">
          {ta
            ? '1947–1955 வரையிலான தமிழ் கலை இலக்கிய இதழ்'
            : 'A Tamil literary magazine, 1947–1955'}
        </p>
      </div>

      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'திராவிட கருத்தியலைப் பட்டித்தொட்டி எங்கும் பரப்பும் முயற்சிக்குத் திராவிட கருத்தியலாளர்கள் பல்வேறு ஊடகங்களைப் கைக்கொண்டனர். அவற்றுள் இதழ்கள் குறிப்பிடத்தக்கன. குடியரசு, விடுதலை, திராவிடநாடு, திராவிடன், போர்வாள், தனியரசு, கிளர்ச்சி, குயில் போன்ற இதழ்கள் மிகப்பெரிய அளவில் அறிவு அரசியல் தளத்தில் தமிழ் மக்களிடையே பெரும் தாக்கத்தை ஏற்படுத்தின. இவ்விதழ்கள் பகுத்தறிவு, சுயமரியாதை, சமத்துவம் போன்ற கொள்கைகளை மக்களிடையே பரப்பியதுடன், சாதி, மத மூடநம்பிக்கைகளுக்கு எதிரான கருத்துக்களை மிகக் காத்திரமாக முன்வைத்தன.'
            : 'Dravidian ideologues employed various media to spread their philosophy far and wide, with magazines being particularly significant. Publications such as Kudiyarasu, Viduthalai, Dravidanadu, Dravidan, Porval, Thaniyarasu, Kilarchi, and Kuyil made a tremendous impact on Tamil society in the intellectual and political arena. These magazines propagated the principles of rationalism, self-respect, and equality among the people, while powerfully challenging caste and religious superstitions.'}
        </div>

        <div className="about-text">
          {ta
            ? "1900களில் வெளிவந்த இதழ்கள் சமூக மாற்றத்திற்கும் முன்னேற்றத்திற்கும் பெருந்துணையாக அமைந்துள்ளன என்பது வரலாற்று ரீதியான உண்மை. 1947 முதல் 1955 வரை இயங்கிய கலை இலக்கிய இதழ் 'பொன்னி'. பொன்னி இதழ் திரு. அரு. பெரியண்ணன் மற்றும் திரு. முருகு. சுப்பிரமணியம் ஆகியோரால் 1947ஆம் ஆண்டு பிப்ரவரி மாதம் தொடங்கப்பெற்றது. தொடங்கப்பட்ட முதல் வருடத்தில் மாதம் ஓர் இதழ் என வெளிவந்த பொன்னி 1948 முதல் மாதம் ஈரிதழாக வெளிவந்தது."
            : "It is a historical fact that magazines published from the 1900s served as a great aid to social change and progress. Ponni was an arts and literary magazine that ran from 1947 to 1955. It was founded by Mr. Aru. Periyannan and Mr. Murugu. Subramaniam in February 1947. In its first year, Ponni was published as a monthly magazine, and from 1948 onward it became a bi-monthly publication."}
        </div>
      </div>

      {/* ── Image 1 ── */}
      <div className="about-image">
        <img
          src={`${API_BASE}/api/images/about/about1.jpg`}
          alt={ta ? 'பொன்னி இதழ்' : 'Ponni Magazine'}
          onError={(e) => { e.target.style.display = 'none'; }}
        />
      </div>

      <hr className="about-divider" />

      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'திராவிட இதழ்களின் வரிசையில் வைத்து போற்றத்தக்க பெரிதும் அறியப்படாத இதழாகப் பொன்னி இதழ் திகழ்கிறது. பகுத்தறிவு, சுயமரியாதை, சமத்துவம் ஆகியவற்றை மிகத் தீவிரமாக எடுத்துரைக்கும் இதழாக இவ்விதழ் வெளிவந்தது. தமிழகத்தின் தலைசிறந்த எழுத்தாளர்களும் படைப்பாளர்களும் தம் சீரிய கருத்துக்களை இவ்விதழின்வழி எடுத்துரைத்தனர். தமிழ்ச் சமூகத்தை அறிவுச் சமூகமாக்கும் முன்னெடுப்பில் பொன்னி இதழின் பணி தலையாயதாகும்.'
            : "Among the Dravidian publications, Ponni stands as a remarkable yet largely lesser-known magazine. It was published as a journal that vigorously advocated rationalism, self-respect, and equality. The finest writers and creators of Tamil Nadu expressed their distinguished ideas through this magazine. Ponni's contribution to transforming Tamil society into an intellectually aware community was paramount."}
        </div>

        <div className="about-pull-quote">
          {ta
            ? "'திராவிடர் கழகத்தை ஆதரிக்கும் ஏடுகள் மிகக் குறைவாக இருந்த காலம். அவையும் அழகில்லாமல், அச்சுப்பிழை மிகுந்து வெளிவந்தன. அந்த நேரத்தில் வண்ண முகப்பு அட்டை போட்டு அழகாக நடந்த இதழ் 'பொன்னி' தான்.'"
            : '"There were very few publications supporting the Dravidar Kazhagam at that time. Even those were published without aesthetics and full of printing errors. At that time, the only magazine that came out beautifully with a colour cover page was Ponni."'}
          <cite>&mdash; {ta ? 'கவியரசு கண்ணதாசன்' : 'Poet Laureate Kannadasan'}</cite>
        </div>

        <div className="about-text">
          {ta
            ? "திராவிடக் கருத்தியலை துப்பாக்கியாகச் செயல்பட்ட திரு. அரு. பெரியண்ணன் அவர்களும், உள்வாங்கி இரட்டைக்குழல் திரு. முருகு. சுப்பிரமணியம் அவர்களும் இணைந்து 1947ஆம் ஆண்டு பிப்ரவரி மாதம் பொன்னி இதழைத் தொடங்கினர். பொன்னி இதழ் வண்ண அட்டைப்படத்தில் மிக நேர்த்தியாக வடிவமைக்கப்பட்டு வெளியிடப்பெற்றது. பத்திரிக்கை துறையில் மற்றவர்கள் செய்துகாட்டாத புதுமை எல்லாம் அவர்கள் செய்து காட்டினார்கள். இன்றும் தமிழகத்தில் சிலரை அச்சுக்கலை நிபுணர்கள் என்று தேர்ந்தெடுத்தால், அவர்களில் பெரியண்ணன் மிக முக்கியமானவராக இருப்பார்' என்று குறிப்பிட்டுள்ளார்."
            : "Mr. Aru. Periyannan, who wielded Dravidian ideology like a weapon, and Mr. Murugu. Subramaniam together founded Ponni magazine in February 1947. Ponni was published with beautifully designed colour cover pages. They pioneered innovations in publishing that no one else had achieved. Even today, if one were to select typography experts in Tamil Nadu, Periyannan would be among the most important."}
        </div>

        <div className="about-text">
          {ta
            ? "தமிழ் இலக்கிய உலகில் முக்கியமான கவிஞர் பாரதிதாசன் அவரின் 'குயில்' இதழ் அரசால் தடை செய்யப்பட்ட பிறகு பொன்னியில் எழுதினார். அவரின் கொள்கைகளையும் நடையையும் பின்பற்றி எழுதியவர்களை 'பாரதிதாசன் பரம்பரை கவிஞர்கள்' என்று அறிமுகப்படுத்தியது பொன்னி இதழ்."
            : 'After the government banned poet Bharathidasan\'s magazine "Kuyil", he began writing for Ponni. Ponni magazine introduced writers who followed his principles and style as "Bharathidasan Heritage Poets".'}
        </div>
      </div>

      {/* ── Images 2 & 3 ── */}
      <div className="about-double-image">
        <img
          src={`${API_BASE}/api/images/about/about2.jpg`}
          alt={ta ? 'பொன்னி வரலாறு 1' : 'Ponni History 1'}
          onError={(e) => { e.target.style.display = 'none'; }}
        />
        <img
          src={`${API_BASE}/api/images/about/about3.jpg`}
          alt={ta ? 'பொன்னி வரலாறு 2' : 'Ponni History 2'}
          onError={(e) => { e.target.style.display = 'none'; }}
        />
      </div>

      <hr className="about-divider" />

      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'இவ்விதழில் மாநில சுயாட்சி, இந்தித் திணிப்பு, தனித்தமிழ் பற்று, விடுதலை, அரசியல், சமூகம் சார்ந்த திராவிடச் சிந்தனைகள் கட்டுரைகளாக, கதைகளாக, கவிதைகளாக. கிறனாய்வுகளாக, துணுக்குகளாக வெளிவந்தன.'
            : 'The magazine featured Dravidian thoughts on state autonomy, opposition to Hindi imposition, Tamil language pride, freedom, politics, and social issues — published as articles, stories, poems, literary reviews, and short pieces.'}
        </div>

        <div className="about-text">
          {ta
            ? 'தந்தை பெரியார், பேரறிஞர் அண்ணா, பாவேந்தர், திரு.வி.க., கலைஞர் மு. கருணாநிதி, கா. அப்பாதுரையார், கவியரசு கண்ணதாசன், டி. கே. சீனிவாசன், மு. வ., மு. அண்ணாமலை, கவிஞர் வாணிதாசன், கவிஞர் சுரதா போன்ற பல முதன்மையான இலக்கிய, அரசியல் ஆளுமைகள் பொன்னியில் எழுதியுள்ளனர்.'
            : 'Prominent literary and political figures who contributed to Ponni include Thanthai Periyar, Perarignar Anna, Bharathidasan, Thiru.Vi.Ka., Kalaignar M. Karunanidhi, Ka. Appadurai, Poet Laureate Kannadasan, T.K. Srinivasan, Mu. Va., Mu. Annamalai, poet Vanidasan, and poet Suradha, among many others.'}
        </div>

        <div className="about-text">
          {ta
            ? 'கவிதைகள், சிறுகதைகள், தொடர்கதைகள், நொடிக் கதைகள், நாடகங்கள், பொதுக் கட்டுரைகள், ஆய்வுக் கட்டுரைகள், ஒப்பாய்வுக் கட்டுரைகள், தொடர் கட்டுரைகள், செய்திப் பாட்டு போன்ற இலக்கிய வகைமைகளில் பொன்னியில் படைப்புகள் வெளியாகியுள்ளன. இது மட்டுமன்றி அட்டைப்படக் குறிப்பு, மகளிர் அழகுக் குறிப்புகள், குழந்தை வளர்ப்புமுறை, பொன்னி வாழ்த்துகள், விகடங்கள், சிறுவர் அரங்கம் (சிறுவர் இலக்கியம்) போன்ற படைப்புகளும் இடம்பெற்றுள்ளன.'
            : "Ponni published works in literary genres including poetry, short stories, serialized novels, flash fiction, plays, general articles, research articles, comparative reviews, serial articles, and news songs. Additionally, the magazine featured cover notes, women's beauty tips, child-rearing guidance, greetings, humour columns, and children's literature."}
        </div>
      </div>

      {/* ── Images 4 & 5 ── */}
      <div className="about-double-image">
        <img
          src={`${API_BASE}/api/images/about/about4.jpg`}
          alt={ta ? 'பொன்னி வரலாறு 3' : 'Ponni History 3'}
          onError={(e) => { e.target.style.display = 'none'; }}
        />
        <img
          src={`${API_BASE}/api/images/about/about5.jpg`}
          alt={ta ? 'பொன்னி வரலாறு 4' : 'Ponni History 4'}
          onError={(e) => { e.target.style.display = 'none'; }}
        />
      </div>

      <hr className="about-divider" />

      <div className="about-section">
        <div className="about-text">
          {ta
            ? '1948ல் போராட்டச் செய்தி நாட்குறிப்பு என்னும் தலைப்பில் கா. அப்பாதுரையார் அவர்கள், அக்கால விடுதலைப் போராட்ட நிலவரங்களை பதிவு செய்துள்ளார். எங்கே, யார் எதற்காக கைது செய்யப் படுகிறார்கள்? அவர்களுக்கு என்ன தண்டனை? போன்றவற்றை இப்பகுதியில் காணமுடிகிறது. தமிழ் இலக்கியம் மட்டுமன்றி சீனம், யார் எதற்காகக் கைது செய்யப் உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர். பாரசீகம், ரஷ்ய, கன்னடம், உருது, வடமொழி, தெலுங்கு போன்ற உலக இலக்கியங்களையும் பொன்னியில் அறிமுகம் செய்துள்ளனர்.'
            : 'In 1948, Ka. Appadurai documented the freedom struggle events under the column "Struggle News Diary", recording who was being arrested, where, and for what reason, along with their sentences. Beyond Tamil literature, Ponni also introduced world literature from Chinese, Persian, Russian, Kannada, Urdu, Sanskrit, and Telugu traditions.'}
        </div>

        <div className="about-text">
          {ta
            ? 'நாடக விளம்பரங்கள், புத்தக விளம்பரங்கள், திரைப்பட விளம்பரங்கள், வணிக விளம்பரங்கள் போன்றவை பொன்னி இதழில் இடம் பெற்றுள்ளன. கலையுலகம் என்ற பகுதியின் கீழ் திரைப்படங்கள், நாடகங்களின் விமர்சனங்களை எழுதியுள்ளனர். பொன்னியில் மேலும் ஒரு சிறப்பிற்குரிய விஷயம் அதில் இடம்பெற்றுள்ள படங்கள் மற்றும் ஓவியங்கள். படைப்பின் தலைப்புகளை வரைந்து இதழில் சேர்த்துள்ளனர். புதுமைப்பித்தன் நினைவுகளைப் பற்றி அவரது மனைவி கமலா அவர்கள் பொன்னி இதழில் எழுதியுள்ளார். பொன்னி இதழ் தொடங்கப்பெற்ற காலத்திலிருந்து இந்தி எதிர்ப்பு குறித்தான எழுத்துகள் தொடர்ந்து காத்திரமாக இடம்பெற்றுள்ளது. அறிஞர்களும் மக்களும் இதில் எழுதியுள்ளனர். பொன்னி இதழ் விடுதலை போராட்ட காலகட்டத்தில் வெளியான இதழ் என்பதால், அக்கால அரசியல் சூழ்நிலைகள் மற்றும் சமூக நிலைகள் படைப்புகளில் பிரதிபலிக்கின்றன.'
            : "Ponni featured advertisements for plays, books, films, and commercial products. Under the 'Art World' section, they published reviews of films and plays. A notable feature was its illustrations and artwork — artistic title headers were hand-drawn for each piece. Pudhumaipithan's wife Kamala wrote about her memories of him in Ponni. From its inception, Ponni consistently featured powerful writings against Hindi imposition. Both scholars and common people contributed. As a magazine published during the freedom struggle era, the political and social conditions of the time are reflected throughout its content."}
        </div>

        <div className="about-text">
          {ta
            ? 'மார்க்சியம், பெண்ணியம் போன்ற இசங்களும் பொன்னி இதழில் இடம்பெற்றுள்ளன. இன்றைய தமிழ்நாடு 1947இல் மதராச் மாகாணமாக இருந்தது. 1950ல் அது மெட்ராச் மாநிலமாக மாறியது. இது தொடர்பான கட்டுரைகள் பொன்னியில் இடம்பெற்றுள்ளன. பொன்னியில் பார்ப்பனியத்திற்கு எதிரான கருத்துக்களும் திராவிடத்தை ஆதரிக்கும் கருத்துக்களும் வலுவாகத் தொடர்ந்து இடம்பெற்று வந்திருக்கின்றன. மாநில சுயாட்சி, தனித்தமிழ் போன்றவற்றைக் குறித்தும் பொன்னியில் எழுதப்பட்டுள்ளன. ஓர் இலக்கியம் கருத்துடன் சேர்ந்து காலத்திற்கு ஏற்ப அமைந்தால் மட்டுமே அது நிலைத்து நிற்கும். பொன்னியில் இடம் பெற்றுள்ள படைப்புகளும் அக்காலகட்ட சூழலுக்கு ஏற்ப அமைந்திருக்கின்றன. பொன்னி இதழ் ஒரு கலை இலக்கிய இதழாக மட்டுமின்றி புரட்சி இதழாகவே இருந்திருக்கிறது.'
            : 'Ideologies such as Marxism and feminism also found space in Ponni. Present-day Tamil Nadu was the Madras Presidency in 1947 and became Madras State in 1950 — articles related to this transition were published in Ponni. Anti-Brahminical views and pro-Dravidian sentiments were consistently and strongly featured. Topics such as state autonomy and pure Tamil advocacy were also explored. Literature endures only when it aligns with the ideas and era of its time — and the works published in Ponni were perfectly attuned to the circumstances of that period. Ponni was not merely an arts and literary magazine — it was truly a revolutionary publication.'}
        </div>

        <div className="about-text">
          {ta
            ? '1947 முதல் 1955 வரையிலான தமிழகத்தின் காலக் கண்ணாடியாகப் பொன்னி இதழ் விளங்குகிறது. தொடக்க காலத் திராவிடக் கருத்தியல்களையும், அவை பரப்பப்பெற்ற வடிவங்களையும் முறைகளையும் ஆயும் ஆய்வாளர்களுக்கு மிகச் சிறந்த களமாகப் பொன்னி இதழ்கள் அமையும். 1947க்கு பிறகான எழுத்துருக்கள், சிந்தனைகள், உரிமை முழக்கங்கள், கேலிச் சித்திரங்கள், நூலறிமுகங்கள் போன்றவற்றை அறியவும், அவற்றை ஆய்வுக்குட்படுத்தவும் பெரும் வாய்ப்பை பொன்னி இதழ்கள் ஏற்படுத்திக் கொடுக்கும்.'
            : 'Ponni magazine serves as a mirror of Tamil Nadu from 1947 to 1955. It provides an invaluable resource for researchers studying early Dravidian ideologies, their forms of dissemination, and methods of propagation. The Ponni archives offer a tremendous opportunity to explore and study post-1947 typography, thought movements, rights slogans, caricatures, and book introductions.'}
        </div>
      </div>
    </div>
  );
};

export default About;
