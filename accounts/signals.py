from django.dispatch import receiver
from django.db.models.signals import post_save, pre_save
from . models import User, UserProfile

@receiver(post_save,sender=User)
def post_save_create_profile_receiver(sender,instance, created, **kwargs):
    if created is True:
        UserProfile.objects.create(user=instance)
        print('User Profile is Created')

    else:
        try:
            profile=UserProfile.objects.get(user=instance)
            profile.save()
        except:
            ## CReate the user profile if not exists    
            profile=UserProfile.objects.create(user=instance)
            print('Profile was not existed but I just created one')
        print('User is updated')

@receiver(pre_save, sender = User)
def pre_save_profile_receiver(sender, instance, **kwargs):
    pass

## post_save.connect(post_save_create_profile_receiver,sender=User)