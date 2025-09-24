from django import forms

class UploadForm(forms.Form):
    audio = forms.FileField(help_text='Upload a short WAV file (mono or stereo).')
    apply_base_denoise = forms.BooleanField(initial=True, required=False, label='Soft denoise')
    suppress_triggers = forms.BooleanField(initial=True, required=False, label='Suppress trigger sounds')

    trigger_sensitivity = forms.FloatField(
        initial=0.15, min_value=0.0, max_value=1.0, required=False,
        help_text='Lower = more aggressive suppression.'
    )

    preserve_speech = forms.BooleanField(
        initial=True, required=False, label='Preserve speech (VAD)'
    )
    vacuum_mode = forms.BooleanField(
        initial=True, required=False, label='Vacuum mode (strong background suppression)'
    )

    vacuum_strength = forms.FloatField(
        initial=0.8, min_value=0.0, max_value=1.0, required=False,
        help_text='How strongly to suppress non-speech background.'
    )

    def clean_audio(self):
        f = self.cleaned_data['audio']
        if not f.name.lower().endswith('.wav'):
            raise forms.ValidationError('Пока принимаем только WAV (.wav). MP3 добавим позже.')
        return f
