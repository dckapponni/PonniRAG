import React from 'react';

const History = ({ language }) => {
  const ta = language === 'ta';

  return (
    <div className="about-container">
      <div className="about-header">
        <h2 className="about-heading">
          {ta ? 'பொன்னி தொகுப்பு முதல் நுண்ணறிவு வரை' : 'From Ponni Collection to Intelligence'}
        </h2>
        <p className="about-subtitle">
          {ta
            ? 'அரிய இதழ்களிலிருந்து செயற்கை நுண்ணறிவு களஞ்சியம் வரை'
            : 'From rare magazines to an AI-powered archive'}
        </p>
      </div>

      {/* ── Section 1: Origins ── */}
      <div className="about-section">
        <div className="about-text">
          {ta
            ? '1947 முதல் 1955 வரை இயங்கிய கலை இலக்கிய அரசியல் வரலாற்று இதழ் பொன்னி. இவ்விதழை நடத்தியவர்கள் அரு. பெரியண்ணன் மற்றும் முருகு. சுப்பிரமணியம் அவர்கள். அரு. பெரியண்ணன் அவர்களின் மகள்வழி பெயரனும் டிசிகாப் நிறுவனத்தின் நிறுவனர் மற்றும் தலைமைச் செயல் அதிகாரியுமான திரு. கார்த்திக் சிதம்பரம் அவர்கள் 2025-ல் பெரியண்ணன் மற்றும் அவர் மனைவி மீனாட்சி அவர்களின் நூற்றாண்டு விழாவைச் சிறப்பிக்கும் விதமாக அவர்களை குறித்து நூல் ஒன்றை எழுத முனைந்தார்.'
            : 'Ponni was an arts, literary, political, and historical magazine that ran from 1947 to 1955. It was published by Aru. Periyannan and Murugu. Subramaniam. Mr. Karthik Chidambaram, the grandson of Aru. Periyannan (through his daughter) and founder and CEO of DCKAP, set out in 2025 to write a book about Periyannan and his wife Meenakshi to mark their centenary celebration.'}
        </div>

        <div className="about-text">
          {ta
            ? 'சிறுவயதிலிருந்து செவி வழியாக பொன்னி இதழ் குறித்து கேள்விப்பட்டிருந்தாலும் அதில் அவர் அதிக கவனம் செலுத்தவில்லை. 2025, ஜனவரியில் புத்தகப் பணிக்காக டெக்சஸ் அமெரிக்காவில் உள்ள பெரியண்ணன் அவர்களின் தலைமகன் பெரி. அழகப்பன் வீட்டிற்கு சென்றார் கார்த்திக் சிதம்பரம். பெரியண்ணன் மறைந்த பின்பு அவர் வாழ்ந்த சேத்துப்பட்டு வீட்டில் நிறைய ஆவணங்கள், பத்திரங்கள், கடிதங்கள், தொழில் முறை கோப்புகள் ஆகியவற்றை அவரின் மகன்கள் கண்டறிந்தனர். அவை அனைத்தும் கொஞ்சம் கொஞ்சமாக பெரி. அழகப்பன் அவர்களுடன் அமெரிக்கா சென்றன. இதில் மிக முக்கியமாக கவனிக்க வேண்டியது பொன்னி இதழின் 85% பதிப்புகளை பல்வேறு இடங்களிலிருந்து கொஞ்சம் கொஞ்சமாகச் சேகரித்து பெரி. அழகப்பன் அவர்கள் பத்திரப்படுத்தி வைத்திருந்தார். கார்த்திக் சிதம்பரம் அவர்கள் அதில் சில இதழ்களைப் படித்துப் பார்க்கும்போது அவற்றில் தந்தை பெரியார், பேரறிஞர் அண்ணா, கலைஞர் மு. கருணாநிதி, பாவேந்தர் பாரதிதாசன், கவியரசு கண்ணதாசன், நாவலர் நெடுஞ்செழியன், கா. அப்பாத்துரையார், சுரதா, டி. கே. சீனிவாசன், மு. அண்ணாமலை, மு. வ., கவிஞர் வாணிதாசன், திரு. வி. க. போன்ற பல ஆளுமைகள் எழுதியிருந்தனர்.'
            : 'Although Karthik Chidambaram had heard about Ponni magazine since childhood, he had not paid much attention to it. In January 2025, he visited the home of Periyannan\'s eldest son Peri. Azhagappan in Texas, USA, for the book project. After Periyannan\'s passing, his sons had discovered numerous documents, deeds, letters, and professional files at his Chetpet home. These gradually made their way to the USA with Peri. Azhagappan. Notably, Azhagappan had painstakingly collected and preserved 85% of all Ponni magazine editions from various sources. When Karthik Chidambaram read through some of these issues, he found contributions from prominent figures such as Thanthai Periyar, Perarignar Anna, Kalaignar M. Karunanidhi, Bharathidasan, Kannadasan, Navalar Nedunchezhiyan, Ka. Appadurai, Suradha, T.K. Srinivasan, Mu. Annamalai, Mu. Va., poet Vanidasan, and Thiru. Vi. Ka.'}
        </div>
      </div>

      <hr className="about-divider" />

      {/* ── Section 2: Discovery & Research ── */}
      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'அந்தக் காலக்கட்டத்தின் அரசியல் மற்றும் சமூகச் சூழலை அறிந்து கொள்வதற்கான வரலாற்று ஆவணமாக பொன்னி இதழ்கள் தோன்றின. மாமா அழகப்பன் வாயிலாக பொன்னி இதழ்கள் குறித்து ஆய்வுகள் மேற்கொண்டு \'பொன்னி ஆசிரியவுரைகள்\' என்று நூலை வெளியிட்ட பேராசிரியர் மு. இளங்கோவன் அவர்கள் குறித்தும் கார்த்திக் சிதம்பரம் அறிந்துகொண்டு அவரிடமும் தொடர்பு கொண்டு பேசி பல தகவல்களை அறிந்து கொண்டார்.'
            : 'The Ponni magazines emerged as historical documents for understanding the political and social landscape of that era. Karthik Chidambaram also learned about Professor Mu. Ilangovan, who had conducted research on Ponni magazines through (uncle) Azhagappan and published the book "Ponni Editorials", and contacted him to gather further information.'}
        </div>

        <div className="about-text">
          {ta
            ? 'பொன்னி இதழ்களின் எழுத்துச் சாரம் மற்றும் தீவிர திராவிட அரசியலை உணர்ந்த கார்த்திக் சிதம்பரம் அவர்கள் இதனை எண்ம வடிவில் (digital) வெளிக்கொணர்ந்தால் இன்னும் நூறு ஆண்டுகள் கழிந்தாலும் ஒரு வரலாற்று சுவடாக இருக்கும் என்று நினைத்தார். 2025, மே மாத அமெரிக்க பயணத்தின் போது அழகப்பன் அவர்கள் சேகரித்து வைத்திருந்த அனைத்து இதழ்களையும் சென்னைக்குக் கொண்டு வந்தார்.'
            : 'Recognizing the literary essence and intense Dravidian politics in the Ponni magazines, Karthik Chidambaram envisioned that digitizing them would preserve them as historical records for at least another hundred years. During his trip to the USA in May 2025, he brought all the issues collected by Azhagappan back to Chennai.'}
        </div>
      </div>

      <hr className="about-divider" />

      {/* ── Section 3: Digitization ── */}
      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'அமெரிக்காவில் உள்ள சிகாகோ பல்கலைக்கழகத்தால் பாதுகாக்க பெற்று மிகச் சிறப்பாகச் சென்னையில் தனித்து இயங்கி வரும் ரோஜா முத்தையா நூலகத்தை கார்த்திக் சிதம்பரம் தொடர்பு கொண்டார். அங்கு பணியாற்றும் சு. முத்துமாலதி அவர்களிடம் பொன்னி இதழ் குறித்த விவரங்களைக் கேட்டறிந்தார். 1940-களில் வெளிவந்த பொன்னியின் ஓர் இதழை அவர்கள் டிஜிட்டல் வடிவில் பாதுகாத்து வைத்திருந்தனர். இது மட்டுமின்றி தமிழ் இணையக் கல்விக் கழகத்திலும் பொன்னியின் சில இதழ்கள் இருப்பதாக கூறினார்கள். ரோஜா முத்தையா நூலக இயக்குனர் திரு. சுந்தர் கணேசன் அவர்களை சந்தித்து பொன்னியின் அனைத்து இதழ்களையும் பாதுகாக்க வேண்டிய நடவடிக்கைகளை மேற்கொள்ள துவங்கினார் கார்த்திக் சிதம்பரம். சுந்தர் கனேசன் அவர்கள் அறிமுகப்படுத்திய ரோஜா முத்தையா நூலகத்தின் துணை இயக்குனர் மணிகண்ட சுப்பு மற்றும் குழு, பொன்னி இதழ்களைப் பாதுகாப்பதில் பெரும் பங்காற்றினர். இவ்வாறாக ரோஜா முத்தையா நூலகத்தின் உதவியுடன் பொன்னி இதழ்கள் எண்ம இதழ்களாக பாதுகாக்க பெற்றன.'
            : 'Karthik Chidambaram contacted the Roja Muthiah Research Library in Chennai, which is preserved by the University of Chicago. He inquired about Ponni magazine details with Su. Muthumalathi who worked there. They had preserved one issue of Ponni from the 1940s in digital format. They also mentioned that some Ponni issues were available at the Tamil Nadu Virtual Academy. Meeting with library director Mr. Sundar Ganesan, Karthik Chidambaram began taking steps to preserve all Ponni issues. Deputy director Manikanda Subbu and his team, introduced by Sundar Ganesan, played a major role in preserving the Ponni magazines. Thus, with the help of the Roja Muthiah Research Library, the Ponni magazines were preserved in digital format.'}
        </div>

        <div className="about-pull-quote">
          {ta
            ? 'எண்ம இதழ்களை ரோஜா முத்தையா ஆராய்ச்சி நூலகம், தமிழ்நாடு இணைய கல்விக் கழகம் (Tamilnadu Virtual Academy), Project Madurai, Internet Archive போன்ற தளங்களில் பதிவிறக்கம் செய்து கொள்ளலாம்.'
            : 'The digitized magazines can be downloaded from platforms such as the Roja Muthiah Research Library, Tamil Nadu Virtual Academy, Project Madurai, and Internet Archive.'}
        </div>
      </div>

      <hr className="about-divider" />

      {/* ── Section 4: Compilation & Book ── */}
      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'இந்த முயற்சிகள் நடந்து கொண்டிருந்த போது சென்னைக் கிறித்தவக் கல்லூரியின் தமிழ்த்துறை உதவிப் பேராசிரியர் முனைவர் சா. கருணாகரன் அவர்கள் LinkedIn செயலி மூலமாக அறிமுகமானார். அவரிடம் தொடர்பு கொண்டு பேசிய போது தந்தை பெரியார் அவர்கள் நடத்திய \'குடியரசு\' இதழ்கள் தொகுப்பை எடுத்துக்காட்டாகக் கூறி \'பொன்னி இதழ்கள் அனைத்தையும் தொகுத்து, தொகுப்புகளாக வெளியிட வேண்டும்\' என்று அறிவுறுத்தினார். இதனால் ஆய்வு மாணவர்கள், திராவிட கருத்தியலாளர்கள் என்று பலருக்கு பெரும் பயனாக இருக்கும் என்று கூறினார்.'
            : 'During these efforts, Dr. Sa. Karunakaran, Assistant Professor of Tamil at Madras Christian College, connected through LinkedIn. Citing the compilation of Thanthai Periyar\'s "Kudiyarasu" magazines as an example, he advised that all Ponni issues should be compiled and published as collections. He noted that this would greatly benefit research students, Dravidian ideologists, and many others.'}
        </div>

        <div className="about-text">
          {ta
            ? 'சென்னைக் கிறித்தவக் கல்லூரி மாணவர்கள் அ. வர்ஷினி, ச. லக்ஷ்யா ஸ்ரீ, ர. யோகேஷ்வரி, பா. சண்முக பிரியா, அ. மெல்வினா, ரா. ஆகாஷ், ந. ஹரிஷ், கா. உமாதேவி, ச. சபரிஹா, ர. கார்த்திகேயன், பூ. நந்தன், ஆ. பெனிஷா, செ. அருள், செ. சௌமியா ஸ்ரீ ஆகியோரின் துணை கொண்டு பொன்னி இதழ்களின் தொகுப்புப் பணி தொடங்கப் பெற்றது. முதல் கட்டமாக ஒளி எழுத்துணறி (OCR) கோப்புகளாகப் பொன்னி இதழ்களை மாற்ற வேண்டும். பொன்னி இதழில் இருக்கும் பழைய எழுத்துக்களாலும் பெரிய அளவு கோப்புகளாலும் பொதுச் செயலிகள் அல்லது தளங்கள் தேர்ந்த முறையில் எழுத்துக்களைக் கொடுக்கவில்லை. இந்தக் கட்டத்தில் Tesseract மற்றும் Google Vision திறந்த மூல மென்பொருள் பயன்படுத்தப்பட்டு PDF கோப்புகளாக இருந்த பொன்னி இதழ்கள் எழுத்து வடிவ கோப்புகளாக மாற்றப்பட்டது. எனினும் எழுத்துக் கோப்புகளில் அதிக பிழைகள் இருந்தன. மாணவர்களால் வார்த்தைக்கு வார்த்தை பிழை திருத்தம் செய்யப்பெற்றது.'
            : 'With the help of Madras Christian College students — A. Varshini, S. Lakshya Sri, R. Yogeshwari, Pa. Shanmuga Priya, A. Melvina, Ra. Aakash, N. Harish, Ka. Umadevi, S. Sabariha, R. Karthikeyan, Pu. Nandhan, A. Benisha, Se. Arul, and Se. Sowmiya Sri — the compilation work of Ponni magazines began. The first phase required converting the Ponni magazines into OCR files. Due to the old typefaces and large file sizes, standard applications and platforms could not accurately recognize the text. At this stage, Tesseract and Google Vision open-source software were used to convert the PDF files into text documents. However, the text files contained numerous errors, which were painstakingly corrected word by word by the students.'}
        </div>
      </div>

      <hr className="about-divider" />

      {/* ── Section 5: Centenary Event ── */}
      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'பெரியண்ணன் - மீனாட்சி நூற்றாண்டு விழாவில் பொன்னி இதழ்களை அறிமுகம் செய்யும் விதமாக \'பொன்னி இதழ்த் தொகுப்பு அறிமுகம் 1947 - 1955\' என்று நூலை உருவாக்க குழு திட்டமிட்டது. இதனுடன் \'பெரியண்ணன் அறியப்படாத வரலாறு\' என்ற பெரியண்ணன் மற்றும் அவர் குடும்பத்தினர் குறித்த நூல் பணியும் மேற்கொள்ளப்பட்டது. இந்நிகழ்வை ஒரு கொண்டாட்ட விழாவாக மட்டும் நடத்தாமல் கல்லூரி மாணவர்களுக்கான ஒரு நாள் சிறப்புக் கருத்தரங்கமாக நடத்தினர். பல்வேறு கல்லூரிகளிலிருந்தும் மாணவர்கள் வந்து பங்குக் கொண்டனர். 25.09.2025 அன்று பெரிய அளவில் நிகழ்வு ஏற்பாடு செய்யப்பட்டு அமைச்சர் திரு. எஸ். ரகுபதி அவர்கள் நூல்களை வெளியிட மாநிலங்களவை உறுப்பினர் திருச்சி என். சிவா அவர்கள் பெற்றுக் கொண்டார். மாண்புமிகு தமிழ்நாடு முதல்வர் அவர்கள் \'பெரியண்ணன் எழுதப்படாத வரலாறு\' நூலிற்கு வாழ்த்துரை வழங்கியிருந்தார்.'
            : 'The team planned to create the book "Ponni Magazine Collection Introduction 1947–1955" to introduce the Ponni magazines at the Periyannan–Meenakshi Centenary Celebration. Alongside this, work on the book "Periyannan: The Unwritten History" about Periyannan and his family was also undertaken. The event was organized not just as a celebration but as a one-day special seminar for college students. Students from various colleges participated. On 25.09.2025, a grand event was organized where Minister Mr. S. Ragupathi released the books, received by Rajya Sabha member Tiruchi N. Siva. The Hon\'ble Chief Minister of Tamil Nadu provided a congratulatory message for the book "Periyannan: The Unwritten History".'}
        </div>

        <div className="about-text">
          {ta
            ? 'இந்நிகழ்விற்கு பின்னர் பிழை திருத்தப் பணி தொடர்ந்தது. பொன்னி இதழ்களை நூல் தொகுப்புகளாக வெளிக்கொண்டு வந்தால் அவை அனைத்து நிலை மக்களிடமும் எளிமையாக சென்று சேர்வது கடினம். எனவே பொன்னி இதழ் தொகுப்புகளை மின் நூலாக (E-Book) வெளியிட குழு திட்டமிட்டது. இதுவரை வெளியான மின் இதழ்கள், மின் நூல்கள் போல் இல்லாமல் பொன்னி தொகுப்புகள் பழைய இதழ் அமைப்பில் புதிய இதழ்களாக வடிவமைக்கப்பட்டு தொகுப்புகளாக மாற்றம் பெற்றன.'
            : 'After the event, the proofreading work continued. The team realized that publishing Ponni magazines as physical book compilations would make it difficult to reach people across all levels easily. Therefore, they planned to publish the Ponni collections as e-books. Unlike typical e-magazines and e-books published previously, the Ponni collections were designed as new editions while retaining the classic magazine layout.'}
        </div>
      </div>

      <hr className="about-divider" />

      {/* ── Section 6: Digital Archive & AI ── */}
      <div className="about-section">
        <h3 className="history-section-title">
          {ta
            ? 'அரிய பொன்னி இதழ்களிலிருந்து திறந்தவெளி மின்னூலகம் வரை: ஆவணப் பாதுகாப்பின் பயணம்'
            : 'From Rare Ponni Magazines to an Open Digital Library: The Journey of Document Preservation'}
        </h3>

        <div className="about-text">
          {ta
            ? 'இதனோடு மட்டும் நின்றுவிடாமல் செயற்கை நுண்ணறிவு காலகட்டத்திற்கு ஏற்றாற்போல பொன்னி குறித்த தகவல்கள் செரிவாகக் கிடைக்கும் வகையில் தேடுதிறன் கொண்ட பொன்னி இதழ்க் களஞ்சியம் (ponniarchive.com) உருவாக்கப் பெற்றது.'
            : 'Not stopping there, in keeping with the era of artificial intelligence, the searchable Ponni Magazine Archive (ponniarchive.com) was created to make information about Ponni richly accessible.'}
        </div>

        <div className="about-text">
          {ta
            ? 'பொன்னி RAG என்பது 1947 முதல் 1955 வரை வெளியான பொன்னி தமிழ் இதழ்களின் உள்ளடக்கங்களை செயற்கை நுண்ணறிவு (AI) உதவியுடன் தேடவும், பகுப்பாய்வு செய்யவும், பயனர்களின் கேள்விகளுக்கு துல்லியமான பதில்களை வழங்கவும் உருவாக்கப்பட்ட ஒரு அறிவார்ந்த Retrieval-Augmented Generation (RAG) எனும் அமைப்பாகும். இதற்காக பொன்னி இதழ்களின் கோப்புகள் பாதுகாப்பான சேவையகங்களில் சேகரிக்கப்பட்டு, பின்னர் ஒவ்வொரு ஆவணத்திலிருந்தும் உரை பிரித்தெடுக்கப்பட்டு, தனித்தனி கட்டுரைகள் அடையாளம் காணப்படுவதுடன், தலைப்பு, ஆசிரியர், இதழ் எண், ஆண்டு போன்ற மேனிலைத் தரவு தகவல்களும் சேகரிக்கப்படுகின்றன.'
            : 'Ponni RAG is an intelligent Retrieval-Augmented Generation (RAG) system built to search, analyze, and provide accurate answers to users\' questions using the contents of Ponni Tamil magazines published from 1947 to 1955, powered by artificial intelligence (AI). For this purpose, the Ponni magazine files were stored on secure servers, text was extracted from each document, individual articles were identified, and metadata such as title, author, issue number, and year were collected.'}
        </div>

        <div className="about-text">
          {ta
            ? 'இந்த கட்டுரைகள் அவற்றின் உள்ளடக்கத்தின் அடிப்படையில் கவிதை, தலையங்கம், இலக்கிய விமர்சனம், அரசியல், வரலாறு, பெண்கள் நலன், சிறுவர் பகுதி உள்ளிட்ட பல்வேறு வகைகளாக தானாக வகைப்படுத்தப்படுகின்றன. பின்னர், ஒவ்வொரு கட்டுரையும் அதன் பொருள் சார்ந்த திசையன் வடிவமாக மாற்றப்பட்டு தரவுதளத்தில் சேமிக்கப்படுகின்றன. இதன் மூலம் ஆயிரக்கணக்கான கட்டுரைகளிலிருந்து தொடர்புடைய தகவல்களை மிக வேகமாகவும் துல்லியமாகவும் கண்டறியக்கூடிய ஒரு வலுவான அறிவுத்தளம் உருவாக்கப்படுகிறது.'
            : 'These articles are automatically categorized based on their content into various types including poetry, editorials, literary criticism, politics, history, women\'s welfare, and children\'s sections. Each article is then converted into a semantic vector representation and stored in a database. This creates a robust knowledge base capable of quickly and accurately finding relevant information from thousands of articles.'}
        </div>

        <div className="about-text">
          {ta
            ? 'பயனர் கேள்வி கேட்கும்போது, சொற்களின் நேரடிப் பொருளைத் தாண்டி, பயனரின் தேடல் நோக்கம் மற்றும் சூழலைப் புரிந்துகொண்டு இந்த அறிவுத்தளம் முடிவுகளை வழங்குகிறது. உள்ளடக்க விளக்கம் அல்லது பொருள் சார்ந்த கேள்விகளுக்கு சொற்பொருள் தேடல் மூலமாகவும், வார்தைகளின் அடிப்படையில் கேட்கப்படும் கேள்விகளுக்கு பாரம்பரிய தேடல் மூலமாகவும் அறிவுத்தளம் பொருத்தமான கட்டுரைகளைத் தேர்ந்தெடுக்கின்றது. இந்த தகவல்கள் கூகுளின் மொழி மாதிரியான Google Gemini 2.5 Flash உதவியுடன் இயல்பான தமிழில் பதில்களாக பெறப்படுகின்றன. அறிவுத்தளத்தின் பதில்களின் தரம் மற்றும் துல்லியம் சில அளவுகோல்கள் மூலம் தொடர்ந்து மதிப்பீடு செய்யப்படுகிறது.'
            : 'When a user asks a question, the knowledge base delivers results by understanding the user\'s search intent and context beyond the literal meaning of words. For content or meaning-based queries, semantic search is used, while keyword-based questions use traditional search to select relevant articles. The information is then presented as natural Tamil responses with the help of Google\'s language model, Google Gemini 2.5 Flash. The quality and accuracy of the knowledge base\'s responses are continuously evaluated through specific metrics.'}
        </div>

        <div className="about-text">
          {ta
            ? 'தமிழ் மற்றும் ஆங்கில மொழிகளில் செயல்படும் அறிவுத்தளம், மொழி மாதிரியின் விதிகள் தாக்குதலுக்கு ஆளாகாத வண்ணம் அதற்கு எதிரான பாதுகாப்பு அம்சங்களையும் கொண்டுள்ளது. மேலும், முந்தைய உரையாடல்களின் சூழலை கருத்தில் கொண்டு தொடர்சியாகக் கேட்கப்படும் கேள்விகளுக்கு அறிவுத்தளத்தால் பதிலளிக்க முடியும். பயனர்கள் செயற்கை நுண்ணறிவு உரையாடல், மின்னூலகம், கட்டுரை வகைகள் போன்ற வசதிகளின் மூலம் பொன்னி இதழ்களின் உள்ளடக்கங்களை எளிதாக உலவலாம்.'
            : 'The knowledge base operates in both Tamil and English languages and includes security features to protect the language model from adversarial attacks. Furthermore, the system can respond to follow-up questions by considering the context of previous conversations. Users can easily browse the contents of Ponni magazines through AI conversations, the digital library, and article categories.'}
        </div>
      </div>

      <hr className="about-divider" />

      {/* ── Section 7: Significance & Closing ── */}
      <div className="about-section">
        <div className="about-text">
          {ta
            ? 'இந்த முயற்சியின் சிறப்பு என்னவென்றால், செயற்கை நுண்ணறிவு இணையத்தில் கிடைக்கும் பொதுவான தகவல்களை அடிப்படையாகக் கொண்டு பதிலளிப்பதில்லை; மாறாக, பொன்னி இதழ்களில் உறுதிப்படுத்தப்பட்ட ஆதாரங்களிலிருந்து மட்டுமே தகவல்களை மீட்டெடுத்து பதில்களை வழங்குகிறது. இதன் மூலம் வரலாற்று உண்மைத்தன்மை, மொழி நுட்பம் மற்றும் பண்பாட்டுச் சூழல் ஆகியவை பாதுகாக்கப்படுகின்றன. பொன்னி RAG, ஒரு பழைய இதழ்த் தொகுப்பின் எண்ம ஆவணக் காப்பகமாக மட்டுமல்லாமல், ஆய்வாளர்கள், மாணவர்கள், வாசகர்கள் மற்றும் பொதுமக்கள் உரையாடல் வழியாக தகவல்கள் அறிந்துகொள்ளக்கூடிய உயிருள்ள அறிவுக் களஞ்சியமாக உருவாக்கப்பட்டிருக்கிறது. மேலும், இந்த முயற்சி எதிர்காலத்தில் குடியரசு, மணிக்கொடி, ஆனந்த விகடன் போன்ற பிற வரலாற்று சிறப்புமிக்க இதழ்களின் ஆவணக் களஞ்சியங்களுக்கும், இந்திய மொழி ஆவணத் தொகுப்புகளுக்கும் விரிவுபடுத்தக்கூடிய ஒரு முன்னோடி மாதிரியாக அமைகிறது.'
            : 'What makes this effort special is that the AI does not rely on generic information available on the internet; instead, it retrieves information exclusively from verified sources within the Ponni magazines. This preserves historical authenticity, linguistic nuance, and cultural context. Ponni RAG has been created not merely as a digital archive of an old magazine collection, but as a living knowledge repository where researchers, students, readers, and the general public can learn through conversation. Moreover, this initiative serves as a pioneering model that can be expanded to other historically significant magazines such as Kudiyarasu, Manikodi, and Ananda Vikatan, as well as Indian language document collections.'}
        </div>

        <div className="about-text">
          {ta
            ? 'பொன்னி என்னும் வரலாற்றுச் சிறப்புமிக்க இதழைக் கண்டறிந்த பின்பு அதனை எடுத்து ஆவண காப்பகத்திடம் மட்டும் ஒப்படைத்து விடாமல் தனியாக குழு அமைத்து, பணம் செலவழித்து, பொன்னி இதழ்களை எளிமையாக மக்கள் அணுகி பயன்பெற தேவையான அனைத்து முன்னெடுப்புகளையும் செய்தார் கார்த்திக் சிதம்பரம். இந்தப் பயணத்தின் தொடக்க முதல் இறுதி வரை ஒருங்கிணைப்பாளராக திறம்பட செயல்பட்டவர் ஸ்ரீஜா சந்தானம் அவர்கள்.'
            : 'After discovering the historically significant Ponni magazine, Karthik Chidambaram did not simply hand it over to an archive. Instead, he formed a dedicated team, invested resources, and took all necessary initiatives to make the Ponni magazines easily accessible to the public. Sreeja Santhanam served as the efficient coordinator throughout this journey from start to finish.'}
        </div>

        <div className="about-pull-quote">
          {ta
            ? 'பல்வேறு சிக்கல்களையும் சவால்களையும் எதிர்கொண்டு புதிய முயற்சிகளை மேற்கொண்டு பொன்னி இதழ்கள் மக்கள் கைகளுக்குள் கொடுக்கப்பட்டிருக்கிறது. திராவிட கருத்தியலாளர்கள், சிந்தனையாளர்கள், இலக்கிய மாணவர்கள், ஆய்வாளர்கள், தீவிர வாசகர்கள் போன்றோருக்கு பொன்னி இதழ்க் களஞ்சியம் பொக்கிஷம் போன்றது. இனி இந்த பொக்கிஷத்தை மிகச் சரியாக பயன்படுத்திக் கொள்வது மட்டுமே நமது வேலை.'
            : 'Overcoming various challenges and obstacles, the Ponni magazines have been placed in the hands of the people through new initiatives. For Dravidian ideologists, thinkers, literary students, researchers, and avid readers, the Ponni Archive is a treasure. Now, our only task is to make the best use of this treasure.'}
        </div>
      </div>
    </div>
  );
};

export default History;
